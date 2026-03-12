import asyncio
import uuid
import time
from typing import Dict, Any, Optional
from source.relay.ollama import OllamaClient
from source.relay.logging import log_event
from source.relay.config import get_settings
from source.relay.session import get_session, append_turn
from source.relay.providers.claude_provider import ClaudeProvider
from source.relay.providers.gemini_provider import GeminiProvider
from source.relay.providers.openai_provider import OpenAIProvider
from source.relay.providers.local_provider import LocalProvider

PROVIDERS = {
    "claude": ClaudeProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
}

SYNTHESIS_PROMPT = """You are the LOCAL DECISION ENGINE of a privacy-preserving AI gateway.

ORIGINAL QUESTION (full context, may contain personal details):
{original_query}

LOCAL AI DIRECT RESPONSE:
{local_answer}

EXTERNAL AI KNOWLEDGE (answered on a depersonalized version — no names or specific identifiers):
{external_answer}

INSTRUCTIONS:
1. The external knowledge provides comprehensive, general analysis — use it as your evidence base.
2. You have the full original context; apply the top guidance to THIS specific situation.
3. Do NOT expose IDs, SSNs, account numbers, or contact details in your answer.
4. Synthesize the strengths of both responses into a single, complete answer.
5. Respond in the SAME LANGUAGE as the original question.

Write the final answer now:"""


class RelayOrchestrator:
    def __init__(self, config=None):
        self.config = config or get_settings()
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
    ) -> Dict[str, Any]:
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        timings = {}

        # ── Load session state ────────────────────────────────────────────────
        # raw_history      : full PII-intact turns (local LLM only)
        # sanitized_history: reformulated turns (safe to pass to external)
        if session_id:
            session = get_session(session_id)
            raw_history = session["raw_history"]
            sanitized_history = session["sanitized_history"]
        else:
            raw_history = []
            sanitized_history = []

        # ── Stage 1: Reformulate ──────────────────────────────────────────────
        # Local LLM sees the sanitized_history (safe prior context) + current raw query.
        # Returns {"route": "hybrid"|"local", "reformulated": str|None}
        t0 = time.time()
        reformat_result = await self.ollama.semantic_reformulate(
            text=user_query,
            sanitized_context=sanitized_history or None,
        )
        route = reformat_result["route"]
        reformulated = reformat_result.get("reformulated") or user_query
        timings["reformulate_ms"] = (time.time() - t0) * 1000
        log_event(trace_id, "reformulate", raw_input=user_query,
                  raw_output={"route": route, "reformulated": reformulated},
                  provider="local", latency_ms=timings["reformulate_ms"])
        print(f"[Stage 1] Route={route} | Reformulated ({len(reformulated)} chars)")

        # ── Build message arrays ──────────────────────────────────────────────
        # Local gets raw history + original query (full context, decision-maker).
        # External gets sanitized history + reformulated query (no PII).
        raw_messages = raw_history + [{"role": "user", "content": user_query}]
        sanitized_messages = sanitized_history + [{"role": "user", "content": reformulated}]

        local_provider = LocalProvider(ollama_client=self.ollama)

        # ── Stage 2: Inference ────────────────────────────────────────────────
        if route == "local":
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
                            sanitized_user=reformulated,
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

        # route == "hybrid": parallel local (raw) + external (sanitized)
        async def _run_local() -> str:
            return await local_provider.generate_async_messages(raw_messages) or ""

        async def _run_external() -> tuple:
            try:
                if hasattr(self.external_provider, "generate_async_messages"):
                    ans = await self.external_provider.generate_async_messages(sanitized_messages) or ""
                elif hasattr(self.external_provider, "generate_async"):
                    ans = await self.external_provider.generate_async(reformulated) or ""
                else:
                    ans = self.external_provider.generate(reformulated) or ""
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
        log_event(trace_id, "external_knowledge", raw_input=reformulated, raw_output=external_answer,
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
                        sanitized_user=reformulated,
                        assistant=final_answer)

        total_latency = (time.time() - start_time) * 1000
        log_event(trace_id, "complete", latency_ms=total_latency)

        return {
            "final_answer": final_answer,
            "steps": {
                "reformulated_query": reformulated,
                "local_answer": local_answer,
                "external_answer": external_answer,
            },
            "provider": self.config.external_provider,
            "timings": timings,
            "latency_ms": total_latency,
            "trace_id": trace_id,
        }
