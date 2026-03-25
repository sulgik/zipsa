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
# The external model answered an anonymized version of the query; this step re-introduces
# the user's specific situation and delivers the final personalized response.
LOCAL_PROMPT = """You are the final step of a local privacy gateway. The original question may contain personal context that was stripped before the external model answered. Your job is to deliver a final response that directly addresses the original question.

ORIGINAL QUESTION:
{original_query}

EXTERNAL ANSWER (produced from an anonymized version of the question):
{external_answer}

INSTRUCTIONS:
1. Use the external answer as your primary source. Apply it to the specific context of the original question.
2. Adjust any generic references to match the user's actual situation where relevant.
3. If the external answer already fully addresses the original question, return it with minimal changes.
4. Do not repeat or surface identifiers (SSNs, account numbers, contact details) in the answer.
5. Do not add privacy warnings, disclaimers, or alerts — the gateway already handles that.
6. Respond in the SAME LANGUAGE as the original question.

Write the final answer now:"""


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

        _, pii_found, _, binding_map = self.safety.scan_and_redact(user_query)
        pii_types = list({k.strip("[]").split("_")[0] for k in binding_map.keys()})

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

        external_query = None
        if execution_path == "hybrid":
            external_query = await self.ollama.reformulate(
                text=user_query,
                conversation_context=sub_thread or None,
            )
            if not external_query:
                execution_path = "local"
                print("[Stage 1] Reformulation failed | Falling back to local-only")
            else:
                _, leaked_pii, _, leaked_binding = self.safety.scan_and_redact(external_query)
                if leaked_pii:
                    execution_path = "local"
                    print(f"[Stage 1] Reformulation leaked PII {list(leaked_binding.keys())[:3]} | Falling back to local-only")
                    external_query = None
                else:
                    print(f"[Stage 1] External query verified clean ({len(external_query)} chars)")

        timings["planning_ms"] = (time.time() - t0) * 1000
        log_event(
            trace_id, "planning", raw_input=user_query,
            raw_output={
                "execution_path": execution_path,
                "external_query": external_query,
                "formal_reason": formal_decision.reason_code,
                "pii_types": pii_types,
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
            return {
                "final_answer": final_answer,
                "steps": {"reformulated_query": None, "local_answer": final_answer, "external_answer": None},
                "provider": "local",
                "timings": timings,
                "latency_ms": total_latency,
                "trace_id": trace_id,
                "pii_detected": pii_found,
                "pii_types": pii_types,
            }

        # ── Stage 2: Hybrid — external inference ─────────────────────────────
        t0 = time.time()
        external_answer, blocked = await _run_external()
        timings["external_ms"] = (time.time() - t0) * 1000
        log_event(trace_id, "external", raw_input=external_query, raw_output=external_answer,
                  provider=self.config.external_provider, latency_ms=timings["external_ms"],
                  safety_flags={"blocked": blocked})
        print(f"[Stage 2] External: {len(external_answer)} chars ({timings['external_ms']:.0f}ms)")

        if blocked or not external_answer:
            t0 = time.time()
            final_answer = await local_provider.generate_async_messages(main_messages) or ""
            timings["local_ms"] = (time.time() - t0) * 1000
            print(f"[Stage 2] Fallback to local: {len(final_answer)} chars")
        else:
            # ── Stage 3: Local — apply external answer to original context ────
            t0 = time.time()
            local_prompt = LOCAL_PROMPT.format(
                original_query=user_query,
                external_answer=external_answer,
            )
            final_answer = await self.ollama.chat(
                [{"role": "system", "content": local_prompt},
                 {"role": "user", "content": "Write the final answer now."}],
                temperature=0.35,
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
        return {
            "final_answer": final_answer,
            "steps": {
                "reformulated_query": external_query,
                "local_answer": None,
                "external_answer": external_answer,
            },
            "provider": self.config.external_provider,
            "timings": timings,
            "latency_ms": total_latency,
            "trace_id": trace_id,
            "pii_detected": pii_found,
            "pii_types": pii_types,
        }
