import asyncio
import uuid
import time
from typing import Dict, Any, Optional
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

SYNTHESIS_PROMPT = """You are the final synthesis step of a local privacy gateway running entirely on the user's machine. Personal details in the original question are kept here locally and were never sent externally — you do not need to warn about them. Your only job is to produce a helpful answer.

ORIGINAL QUESTION:
{original_query}

LOCAL MODEL RESPONSE:
{local_answer}

EXTERNAL KNOWLEDGE (answered on an anonymized version of the question):
{external_answer}

INSTRUCTIONS:
1. Use the external knowledge as the evidence base; apply it to the specific situation in the original question.
2. Do not repeat or surface identifiers (SSNs, account numbers, contact details) in the answer.
3. Do not add privacy warnings, disclaimers, or alerts — the gateway already handles that.
4. Respond in the SAME LANGUAGE as the original question.

Write the final answer now:"""

BINDING_PROMPT = """You are the final binding step of a local privacy gateway. The original question may contain personal context that was stripped before the external model answered. Your job is to take the external answer and apply it back to the original question — filling in any personal specifics, adjusting references, and ensuring the answer directly addresses what the user actually asked.

ORIGINAL QUESTION:
{original_query}

EXTERNAL ANSWER (produced from an anonymized version of the question):
{external_answer}

INSTRUCTIONS:
1. Return the external answer adapted to the original question. Adjust any generic references to match the user's actual context where relevant.
2. Do not repeat or surface identifiers (SSNs, account numbers, contact details) in the answer.
3. Do not add privacy warnings, disclaimers, or alerts.
4. If the external answer already fully addresses the original question without adaptation, return it as-is.
5. Respond in the SAME LANGUAGE as the original question.

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
            # Client sent full messages array (standard OpenAI format)
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
        # 1a. PII scan (deterministic regex — no LLM)
        # 1b. Formal planner: Classify → Propose → Validate (no LLM, always conclusive)
        # 1c. If hybrid, reformulate query for external use (single LLM call)
        t0 = time.time()

        # 1a. PII scan
        _, pii_found, _, binding_map = self.safety.scan_and_redact(user_query)
        pii_types = list({k.strip("[]").split("_")[0] for k in binding_map.keys()})

        # 1b. Formal planner (deterministic, no LLM)
        from source.relay.planner import plan as formal_plan
        formal_decision = formal_plan(
            query=user_query,
            pii_types=pii_types,
            pii_detected=pii_found,
        )
        execution_path = "hybrid" if formal_decision.decision == "hybrid" else "local"
        hybrid_mode = formal_decision.hybrid_mode  # "synthesis" | "selective"
        print(
            f"[Stage 1] Formal planner → {formal_decision.decision} "
            f"(reason={formal_decision.reason_code}, "
            f"mode={hybrid_mode if execution_path == 'hybrid' else 'n/a'}, "
            f"pii={pii_types or 'none'}, "
            f"inj={formal_decision.classifier_tags.injection_risk})"
        )

        # 1c. If hybrid, reformulate for external use (LLM call, sub_thread = external-safe context)
        #     Then verify the reformulation: scan for any PII that leaked through.
        #     If PII is detected in the reformulated query, fall back to local automatically.
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
                _, reformulated_pii_found, _, reformulated_binding = self.safety.scan_and_redact(external_query)
                if reformulated_pii_found:
                    execution_path = "local"
                    leaked = list(reformulated_binding.keys())[:3]
                    print(f"[Stage 1] Reformulation leaked PII {leaked} | Falling back to local-only")
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

        # ── Stage 2: Inference ────────────────────────────────────────────────

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

        if execution_path == "local":
            # Local-only: no external call.
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
            }

        if hybrid_mode == "selective":
            # ── Selective hybrid: reformulate → external → local binding ─────
            # Best for code, text rewrite, structured generation.
            # External answers the anonymized query; local binding step re-applies
            # the original personal context so the answer addresses the actual question.
            t0 = time.time()
            external_answer, blocked = await _run_external()
            timings["external_ms"] = (time.time() - t0) * 1000
            log_event(trace_id, "external_selective", raw_input=external_query,
                      raw_output=external_answer, provider=self.config.external_provider,
                      latency_ms=timings["external_ms"], safety_flags={"blocked": blocked})

            if blocked or not external_answer:
                # External failed — fall back to local
                t0 = time.time()
                final_answer = await local_provider.generate_async_messages(main_messages) or ""
                timings["local_ms"] = (time.time() - t0) * 1000
                print(f"[Stage 2] Selective fallback to local: {len(final_answer)} chars")
            else:
                print(f"[Stage 2] Selective external: {len(external_answer)} chars ({timings['external_ms']:.0f}ms)")
                # Stage 3 (binding): local LLM applies original context to external answer
                t0 = time.time()
                binding_text = BINDING_PROMPT.format(
                    original_query=user_query,
                    external_answer=external_answer,
                )
                final_answer = await self.ollama.chat(
                    [{"role": "system", "content": binding_text},
                     {"role": "user", "content": "Write the final answer now."}],
                    temperature=0.3,
                ) or external_answer
                timings["binding_ms"] = (time.time() - t0) * 1000
                log_event(trace_id, "binding", raw_input=f"ext={external_answer[:100]}",
                          raw_output=final_answer, provider="local",
                          latency_ms=timings["binding_ms"])
                print(f"[Stage 3] Binding: {len(final_answer)} chars ({timings['binding_ms']:.0f}ms)")

            if session_id:
                append_main_turn(session_id, user=user_query, assistant=final_answer)
                if external_query and not blocked:
                    append_sub_turn(session_id, user=external_query, assistant=external_answer)

            total_latency = (time.time() - start_time) * 1000
            log_event(trace_id, "complete", latency_ms=total_latency)
            return {
                "final_answer": final_answer,
                "steps": {"reformulated_query": external_query, "local_answer": None, "external_answer": external_answer},
                "provider": self.config.external_provider,
                "timings": timings,
                "latency_ms": total_latency,
                "trace_id": trace_id,
            }

        # ── Synthesis hybrid: concurrent local + external → local synthesizes ─
        # Best for domain knowledge, analysis — external provides expert knowledge,
        # local applies it to the user's specific personal context.
        async def _run_local() -> str:
            return await local_provider.generate_async_messages(main_messages) or ""

        t0 = time.time()
        local_answer, (external_answer, blocked) = await asyncio.gather(
            _run_local(), _run_external()
        )
        parallel_ms = (time.time() - t0) * 1000
        timings["local_ms"] = parallel_ms
        timings["external_ms"] = parallel_ms
        log_event(trace_id, "local_direct", raw_input=user_query, raw_output=local_answer,
                  provider="local", latency_ms=parallel_ms)
        log_event(trace_id, "external_knowledge", raw_input=external_query, raw_output=external_answer,
                  provider=self.config.external_provider, latency_ms=parallel_ms,
                  safety_flags={"blocked": blocked})
        print(
            f"[Stage 2] Synthesis — local: {len(local_answer)} chars | "
            f"external: {len(external_answer)} chars ({parallel_ms:.0f}ms)"
        )

        # ── Stage 3: Local synthesis ──────────────────────────────────────────
        t0 = time.time()
        synthesis_text = SYNTHESIS_PROMPT.format(
            original_query=user_query,
            local_answer=local_answer,
            external_answer=external_answer,
        )
        final_answer = await self.ollama.chat(
            [{"role": "system", "content": synthesis_text},
             {"role": "user", "content": "Write the final answer now."}],
            temperature=0.4,
        )
        timings["synthesis_ms"] = (time.time() - t0) * 1000
        log_event(
            trace_id, "synthesis",
            raw_input=f"local={local_answer[:100]}|ext={external_answer[:100]}",
            raw_output=final_answer, provider="local",
            latency_ms=timings["synthesis_ms"],
        )

        if session_id:
            append_main_turn(session_id, user=user_query, assistant=final_answer)
            if external_query:
                append_sub_turn(session_id, user=external_query, assistant=external_answer)

        total_latency = (time.time() - start_time) * 1000
        log_event(trace_id, "complete", latency_ms=total_latency)

        return {
            "final_answer": final_answer,
            "steps": {
                "reformulated_query": external_query,
                "local_answer": local_answer,
                "external_answer": external_answer,
            },
            "provider": self.config.external_provider,
            "timings": timings,
            "latency_ms": total_latency,
            "trace_id": trace_id,
        }
