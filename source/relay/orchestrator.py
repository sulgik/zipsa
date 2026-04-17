import asyncio
import uuid
import time
from typing import Dict, Any
from source.relay.ollama import OllamaClient
from source.relay.logging import log_event
from source.relay.config import get_settings
from source.relay.session import get_session, append_main_turn, append_sub_turn
from source.relay.safety import SafetyGuard
from source.relay.providers.claude_provider import ClaudeProvider
from source.relay.providers.gemini_provider import GeminiProvider
from source.relay.providers.openai_provider import OpenAIProvider
from source.relay.providers.local_provider import LocalProvider

PROVIDERS = {
    "anthropic": ClaudeProvider,
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
}

# Stage 3: local LLM applies the external answer back to the original personal context.
# Uses placeholder-based grounding: external LLM sees [NAME_1], [PHONE_1] etc.
# Local LLM restores real values and delivers a personalized final answer.
LOCAL_PROMPT = """Restore placeholders in the external answer using the mapping below. Reply in the same language as the original question. No disclaimers.

Original: {original_query}
Mapping: {binding_table}
External answer: {external_answer}

Final answer:"""


def _is_satisfying(answer: str) -> bool:
    """Heuristic: is the external answer substantive enough to use?"""
    if not answer or len(answer.strip()) < 50:
        return False
    lower = answer.lower()
    refusal_signals = [
        "i cannot ", "i can't ", "i'm unable", "i am unable",
        "as an ai, i", "i don't have access", "i'm not able to",
        "cannot assist", "can't assist", "unable to assist",
        "i apologize, but i cannot", "i'm sorry, but i cannot",
    ]
    return not any(sig in lower for sig in refusal_signals)


class RelayOrchestrator:
    def __init__(self, config=None):
        self.config = config or get_settings()
        self.safety = SafetyGuard()
        self.ollama = OllamaClient(
            model=self.config.local_model,
            host=self.config.local_host,
        )
        provider_cls = PROVIDERS.get(self.config.external_provider)
        if provider_cls is None:
            raise ValueError(
                f"Unknown external provider: {self.config.external_provider!r}. "
                f"Must be one of: {list(PROVIDERS)}"
            )
        import inspect
        from source.auth.token_store import get_token_store
        sig = inspect.signature(provider_cls.__init__)
        if "token_store" in sig.parameters:
            self.external_provider = provider_cls(token_store=get_token_store())
        else:
            self.external_provider = provider_cls()

    async def process_request(
        self,
        user_query: str,
        api_key: str = None,
        session_id: str = None,
        messages_history: list = None,
    ) -> Dict[str, Any]:
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        timings = {}
        trace: list[str] = []

        # Trace: Query received
        query_preview = user_query[:50] + "..." if len(user_query) > 50 else user_query
        trace.append(f"Query received: {query_preview}")

        # ── Load session state ────────────────────────────────────────────────
        # main_thread: full PII-intact local conversation
        # sub_thread : external-safe auxiliary conversation for hybrid turns only
        if session_id:
            session = get_session(session_id)
            main_thread = session["main_thread"]
            sub_thread = session["sub_thread"]
        elif messages_history:
            main_thread = messages_history
            sub_thread = []
            for m in messages_history:
                if m.get("role") == "user":
                    redacted, _, _, _ = self.safety.scan_and_redact(m["content"])
                    sub_thread.append({"role": "user", "content": redacted})
                else:
                    sub_thread.append(m)
        else:
            main_thread = []
            sub_thread = []

        # ── Stage 1: Formal route planning ───────────────────────────────────
        # 1a. PII scan — deterministic regex
        # 1b. Formal planner: Classify → Propose → Validate (no LLM)
        # 1c. If hybrid, reformulate for external use (LLM call) + PII verify output
        t0 = time.time()

        redacted_query, pii_found, _, binding_map = self.safety.scan_and_redact(user_query)
        # Also detect names in "Role: Name" context (e.g. "Patient: John D.")
        redacted_query = self.safety.scan_names_in_context(redacted_query, binding_map)
        pii_types = list({k.strip("[]").split("_")[0] for k in binding_map.keys()})

        # Trace: PII scan result
        if pii_found:
            trace.append(f"PII scan: detected {len(binding_map)} items ({', '.join(pii_types)})")
        else:
            trace.append("PII scan: clean")

        from source.relay.planner import plan_async
        formal_decision = await plan_async(
            query=user_query,
            pii_types=pii_types,
            pii_detected=pii_found,
            ollama_client=self.ollama,
        )
        execution_path = "hybrid" if formal_decision.decision == "hybrid" else "local"
        print(
            f"[Stage 1] → {formal_decision.decision} "
            f"(reason={formal_decision.reason_code}, "
            f"task={formal_decision.classifier_tags.task_type}, "
            f"pii={pii_types or 'none'}, "
            f"inj={formal_decision.classifier_tags.injection_risk})"
        )

        # Trace: Router decision
        trace.append(f"Router decision: {execution_path}")

        external_query = None
        if execution_path == "hybrid":
            # Use pre-redacted query for reformulation so PII is already replaced with placeholders
            external_query = await self.ollama.reformulate(
                text=redacted_query,
                conversation_context=sub_thread or None,
            )
            if not external_query:
                execution_path = "local"
                trace.append("Router decision: local")  # Update trace after fallback
                print("[Stage 1] Reformulation failed | Falling back to local-only")
            else:
                # Trace: Reformulated query
                reformulated_preview = external_query[:80] + "..." if len(external_query) > 80 else external_query
                trace.append(f"Reformulated: {reformulated_preview}")
                _, leaked_pii, _, leaked_binding = self.safety.scan_and_redact(external_query)
                if leaked_pii:
                    execution_path = "local"
                    trace.append("Router decision: local")  # Update trace after PII leak fallback
                    print(f"[Stage 1] Reformulation leaked PII {list(leaked_binding.keys())[:3]} | Falling back to local-only")
                    external_query = None
                else:
                    print(f"[Stage 1] External query verified clean ({len(external_query)} chars): {external_query[:120]}")

        harm_categories = formal_decision.classifier_tags.harm_categories
        timings["planning_ms"] = (time.time() - t0) * 1000
        log_event(
            trace_id, "planning", raw_input=user_query,
            raw_output={
                "execution_path": execution_path,
                "external_query": external_query,
                "formal_reason": formal_decision.reason_code,
                "pii_types": pii_types,
                "harm_categories": harm_categories,
                "injection_risk": formal_decision.classifier_tags.injection_risk,
            },
            provider="formal_planner", latency_ms=timings["planning_ms"],
        )

        # ── Build message arrays ──────────────────────────────────────────────
        main_messages = main_thread + [{"role": "user", "content": user_query}]
        sub_messages = sub_thread.copy()
        if external_query:
            sub_messages.append({"role": "user", "content": external_query})

        local_provider = LocalProvider(ollama_client=self.ollama)

        async def _run_external() -> tuple[str, bool]:
            try:
                if hasattr(self.external_provider, "generate_async_messages"):
                    ans = await self.external_provider.generate_async_messages(sub_messages) or ""
                elif hasattr(self.external_provider, "generate_async"):
                    ans = await self.external_provider.generate_async(external_query or "") or ""
                else:
                    ans = self.external_provider.generate(external_query or "") or ""
                return ans, False
            except Exception as e:
                print(f"[External Error] {e}")
                return f"(External provider failed: {e})", True

        # ── Stage 2: Local-only ───────────────────────────────────────────────
        if execution_path == "local":
            t0 = time.time()
            final_answer = await local_provider.generate_async_messages(main_messages) or ""
            timings["local_ms"] = (time.time() - t0) * 1000
            log_event(trace_id, "local_only", raw_input=user_query, raw_output=final_answer,
                      provider="local", latency_ms=timings["local_ms"])
            print(f"[Stage 2] Local-only: {len(final_answer)} chars")

            if session_id:
                append_main_turn(session_id, user=user_query, assistant=final_answer)

            total_latency = (time.time() - start_time) * 1000
            log_event(trace_id, "complete", latency_ms=total_latency)

            # Trace: Provider and completion
            trace.append("Provider: local")
            trace.append(f"Completed in {int(total_latency)}ms")

            return {
                "final_answer": final_answer,
                "steps": {"reformulated_query": None, "local_answer": final_answer, "external_answer": None},
                "provider": "local",
                "timings": timings,
                "latency_ms": total_latency,
                "trace_id": trace_id,
                "pii_detected": pii_found,
                "pii_types": pii_types,
                "harm_categories": harm_categories,
                "presidio_baseline": redacted_query if pii_found else None,
                "trace": trace,
            }

        # ── Stage 2: Hybrid — parallel external + local prefetch ─────────────
        # Local inference starts immediately alongside the external call.
        # If external is blocked or unsatisfying, local_prefetch is used directly,
        # avoiding a second sequential local call.
        async def _run_local_prefetch() -> str:
            return await local_provider.generate_async_messages(main_messages) or ""

        t0 = time.time()
        (external_answer, blocked), local_prefetch = await asyncio.gather(
            _run_external(),
            _run_local_prefetch(),
        )
        parallel_ms = (time.time() - t0) * 1000
        timings["external_ms"] = parallel_ms  # wall time (ran in parallel)
        timings["local_prefetch_ms"] = parallel_ms
        log_event(trace_id, "external", raw_input=external_query, raw_output=external_answer,
                  provider=self.config.external_provider, latency_ms=timings["external_ms"],
                  safety_flags={"blocked": blocked})
        print(f"[Stage 2] Parallel: external={len(external_answer)}ch local={len(local_prefetch)}ch ({parallel_ms:.0f}ms)")

        if blocked or not external_answer:
            # External failed — use already-computed local prefetch (free, no extra call)
            final_answer = local_prefetch
            timings["local_ms"] = 0.0  # already captured in local_prefetch_ms
            print(f"[Stage 2] Fallback to local prefetch: {len(final_answer)} chars")
        elif not _is_satisfying(external_answer):
            # Stage 3 agent: external answer is a refusal or too vague → use local prefetch
            final_answer = local_prefetch
            timings["local_ms"] = 0.0
            log_event(trace_id, "stage3_agent", raw_input=external_answer[:120],
                      raw_output="use_local", provider="local", latency_ms=0)
            print(f"[Stage 3] Agent: external unsatisfying → using local prefetch ({len(final_answer)} chars)")
        else:
            # Strip leading disclaimer/warning paragraphs the external model may have injected.
            # A paragraph is considered a "warning" if it contains refusal or privacy language
            # but does NOT contain substantive domain content (medications, procedures, etc.).
            import re as _re
            _warn_re = _re.compile(
                r'(?i)('
                r'cannot\s+provide\s+(specific\s+)?(medical|legal|financial)\s+advice|'
                r'important\s+(notice|disclaimer)|privacy\s+(warning|notice|alert|law)|'
                r'must\s+never\s+be\s+shared|remove\s+(this|personal)\s+information|'
                r'please\s+remove\s+PII|never\s+share.*SSN|SSN.*should\s+never|'
                r'not\s+a\s+licensed\s+(healthcare|medical)|for\s+educational\s+purposes\s+only'
                r')',
            )
            _content_re = _re.compile(
                r'(?i)(metformin|sglt2|glp-1|insulin|hba1c|monitoring|interaction|'
                r'revenue|growth|cash\s+flow|q[1-4]\s+|function|def\s+\w+|algorithm)',
            )
            paragraphs = _re.split(r'\n{2,}', external_answer.strip())
            stripped = 0
            while paragraphs and _warn_re.search(paragraphs[0]) and not _content_re.search(paragraphs[0]):
                print(f"[Stage 2] Stripped warning paragraph ({stripped+1}): {paragraphs[0][:60]}...")
                paragraphs.pop(0)
                stripped += 1
            if paragraphs:
                external_answer = '\n\n'.join(paragraphs)
            # ── Stage 3: Local — grounded synthesis with placeholder restoration ──
            t0 = time.time()
            # Build binding table: show only name/entity placeholders (skip raw ID numbers)
            safe_keys = {"NAME", "PERSON", "ORG", "COMPANY", "LOCATION", "DATE", "ROLE"}
            binding_lines = []
            for placeholder, real_value in binding_map.items():
                ptype = placeholder.strip("[]").split("_")[0].upper()
                if ptype in safe_keys:
                    binding_lines.append(f"  {placeholder} → {real_value}")
            binding_table = "\n".join(binding_lines) if binding_lines else "  (no named entities — answer applies as-is)"

            # Run synthesis if: placeholders present in external answer, OR
            # named-entity bindings exist (name/person restoration needed even
            # when the external model referred generically to "the patient").
            import re as _re_ph
            _placeholders_found = _re_ph.findall(
                r'\[(?:NAME|PERSON|ORG|COMPANY|LOCATION|DATE|ROLE)_\d+\]',
                external_answer,
            )
            _has_name_bindings = any(
                placeholder.strip("[]").split("_")[0].upper() in {"NAME", "PERSON"}
                for placeholder in binding_map
            )
            # Pre-substitute: when NAME bindings exist, replace generic "the patient" /
            # "a patient" / "the user" / "the client" references with [NAME] so that
            # Stage 3 can do reliable mechanical substitution instead of relying on
            # the local model to infer the name from the original query.
            if _has_name_bindings:
                _name_key = next(
                    (k for k in binding_map if k.strip("[]").split("_")[0].upper() in {"NAME", "PERSON"}),
                    None,
                )
                if _name_key:
                    import re as _re_generic
                    _generic_re = _re_generic.compile(
                        r'\b(the patient|a patient|the user|a user|the client|a client|the individual|a individual|the person|a person)\b',
                        _re_generic.IGNORECASE,
                    )
                    _pre_subst = _generic_re.sub(_name_key, external_answer)
                    if _pre_subst != external_answer:
                        print(f"[Stage 3] Pre-substituted generic references → {_name_key}")
                        external_answer = _pre_subst
                        _placeholders_found = [_name_key]  # ensure synthesis runs

            if not _placeholders_found and not _has_name_bindings and not pii_found:
                final_answer = external_answer
                timings["local_ms"] = 0.0
                print("[Stage 3] No placeholders, name bindings, or PII — skipping synthesis")
            else:
                local_prompt = LOCAL_PROMPT.format(
                    original_query=user_query,
                    binding_table=binding_table,
                    external_answer=external_answer,
                )
                final_answer = await self.ollama.chat(
                    [{"role": "user", "content": local_prompt}],
                    temperature=0.0,
                ) or external_answer
            timings["local_ms"] = (time.time() - t0) * 1000
            log_event(trace_id, "local_synthesis", raw_input=f"ext={external_answer[:100]}",
                      raw_output=final_answer, provider="local", latency_ms=timings["local_ms"])
            print(f"[Stage 3] Local synthesis: {len(final_answer)} chars ({timings['local_ms']:.0f}ms)")

        if session_id:
            append_main_turn(session_id, user=user_query, assistant=final_answer)
            if external_query and not blocked:
                append_sub_turn(session_id, user=external_query, assistant=external_answer)

        total_latency = (time.time() - start_time) * 1000
        log_event(trace_id, "complete", latency_ms=total_latency)

        # Trace: Provider and completion
        if blocked:
            trace.append("Provider: local")
        else:
            trace.append("Provider: external")
        trace.append(f"Completed in {int(total_latency)}ms")

        return {
            "final_answer": final_answer,
            "steps": {
                "reformulated_query": external_query,
                "local_prefetch": local_prefetch,
                "external_answer": external_answer,
            },
            "provider": self.config.external_provider,
            "timings": timings,
            "latency_ms": total_latency,
            "trace_id": trace_id,
            "pii_detected": pii_found,
            "pii_types": pii_types,
            "harm_categories": harm_categories,
            "presidio_baseline": redacted_query if pii_found else None,
            "trace": trace,
        }
