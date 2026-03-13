import asyncio
import uuid
import time
from typing import Dict, Any, Optional
from source.relay.ollama import OllamaClient
from source.relay.logging import log_event
from source.relay.config import get_settings
from source.relay.session import get_session, append_turn
from source.relay.safety import SafetyGuard
from source.relay.providers.claude_provider import ClaudeProvider
from source.relay.providers.gemini_provider import GeminiProvider
from source.relay.providers.openai_provider import OpenAIProvider
from source.relay.providers.local_provider import LocalProvider

PROVIDERS = {
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
        # raw_history      : full PII-intact turns (local LLM only)
        # sanitized_history: external-safe turns (safe to pass to external)
        if session_id:
            session = get_session(session_id)
            raw_history = session["raw_history"]
            sanitized_history = session["sanitized_history"]
        elif messages_history:
            # Client sent full messages array (standard OpenAI format)
            raw_history = messages_history
            sanitized_history = []
            for m in messages_history:
                if m.get("role") == "user":
                    redacted, _, _, _ = self.safety.scan_and_redact(m["content"])
                    sanitized_history.append({"role": "user", "content": redacted})
                else:
                    sanitized_history.append(m)
        else:
            raw_history = []
            sanitized_history = []

        # ── Stage 1: Route planning ───────────────────────────────────────────
        # Local planner sees the sanitized_history (safe prior context) + current raw query.
        # Returns {"execution_path": "hybrid"|"local", "external_query": str|None}
        t0 = time.time()
        execution_plan = await self.ollama.plan_execution(
            text=user_query,
            sanitized_context=sanitized_history or None,
        )
        execution_path = execution_plan["execution_path"]
        external_query = execution_plan.get("external_query") or user_query
        timings["planning_ms"] = (time.time() - t0) * 1000
        log_event(trace_id, "planning", raw_input=user_query,
                  raw_output={"execution_path": execution_path, "external_query": external_query},
                  provider="local", latency_ms=timings["planning_ms"])
        if execution_path == "local":
            print("[Stage 1] Planner decided local-only | No external call")
        else:
            print(
                f"[Stage 1] Planner decided hybrid | External query prepared "
                f"({len(external_query)} chars)"
            )

        # ── Build message arrays ──────────────────────────────────────────────
        # Local gets raw history + original query (full context, decision-maker).
        # External gets sanitized history + external-safe query (no PII).
        raw_messages = raw_history + [{"role": "user", "content": user_query}]
        sanitized_messages = sanitized_history + [{"role": "user", "content": external_query}]

        local_provider = LocalProvider(ollama_client=self.ollama)

        # ── Stage 2: Inference ────────────────────────────────────────────────
        if execution_path == "local":
            # Local-only: no external call.
            t0 = time.time()
            final_answer = await local_provider.generate_async_messages(raw_messages) or ""
            timings["local_ms"] = (time.time() - t0) * 1000
            log_event(trace_id, "local_only", raw_input=user_query, raw_output=final_answer,
                      provider="local", latency_ms=timings["local_ms"])
            print(f"[Stage 2] Local-only: {len(final_answer)} chars")

            if session_id:
                append_turn(session_id,
                            raw_user=user_query,
                            sanitized_user=external_query,
                            assistant=final_answer)

            total_latency = (time.time() - start_time) * 1000
            log_event(trace_id, "complete", latency_ms=total_latency)
            return {
                "final_answer": final_answer,
                "steps": {
                    "reformulated_query": None,
                    "local_answer": final_answer,
                    "external_answer": None,
                },
                "provider": "local",
                "timings": timings,
                "latency_ms": total_latency,
                "trace_id": trace_id,
            }

        # execution_path == "hybrid": parallel local (raw) + external (sanitized)
        async def _run_local() -> str:
            return await local_provider.generate_async_messages(raw_messages) or ""

        async def _run_external() -> tuple:
            try:
                if hasattr(self.external_provider, "generate_async_messages"):
                    ans = await self.external_provider.generate_async_messages(sanitized_messages) or ""
                elif hasattr(self.external_provider, "generate_async"):
                    ans = await self.external_provider.generate_async(external_query) or ""
                else:
                    ans = self.external_provider.generate(external_query) or ""
                return ans, False
            except Exception as e:
                print(f"[External Error] {e}")
                return f"(External provider failed: {e})", True

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
            f"[Stage 2] Local: {len(local_answer)} chars | "
            f"External: {len(external_answer)} chars ({parallel_ms:.0f}ms)"
        )

        # ── Stage 3: Local synthesis ──────────────────────────────────────────
        # Local LLM combines both answers; external knowledge is evidence only.
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
            append_turn(session_id,
                        raw_user=user_query,
                        sanitized_user=external_query,
                        assistant=final_answer)

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
