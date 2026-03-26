"""
Four evaluation agents for Zipsa comparison.

Agents:
  zipsa           — full Zipsa pipeline (classify → route → reformulate → synthesize)
  local_only      — all queries answered by local LLM, nothing sent to external
  external_only   — raw query sent directly to external LLM, no privacy treatment
  scan_external   — Zipsa regex scanner (SafetyGuard) strips PII, then sends to external
                    (represents mechanical PII-strip baseline, e.g. Presidio-style)

Each returns an AgentResult with:
  external_payload — what was actually sent to the external LLM (None = local only)
  response         — final answer shown to user
  route            — "local" | "external" | "hybrid"
  pii_types        — PII detected in original query
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from .cache import ResponseCache


@dataclass
class AgentResult:
    agent: str
    query: str
    external_payload: Optional[str]  # None = never left local
    response: str
    route: str                        # "local" | "external" | "hybrid"
    pii_types: list = field(default_factory=list)
    latency_ms: float = 0.0
    from_cache: bool = False
    error: Optional[str] = None


# ── Agent 1: Zipsa ────────────────────────────────────────────────────────────

async def run_zipsa(query: str, history: list, orchestrator) -> AgentResult:
    t0 = time.time()
    try:
        result = await orchestrator.process_request(
            user_query=query,
            messages_history=history or None,
        )
        external_payload = result["steps"].get("reformulated_query")
        route = "hybrid" if external_payload else "local"
        return AgentResult(
            agent="zipsa",
            query=query,
            external_payload=external_payload,
            response=result["final_answer"],
            route=route,
            pii_types=result.get("pii_types", []),
            latency_ms=(time.time() - t0) * 1000,
        )
    except Exception as e:
        return AgentResult(
            agent="zipsa", query=query,
            external_payload=None, response="", route="local",
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )


# ── Agent 2: Local-only ───────────────────────────────────────────────────────

async def run_local_only(query: str, history: list, local_provider) -> AgentResult:
    t0 = time.time()
    messages = (history or []) + [{"role": "user", "content": query}]
    try:
        response = await local_provider.generate_async_messages(messages) or ""
        return AgentResult(
            agent="local_only",
            query=query,
            external_payload=None,
            response=response,
            route="local",
            latency_ms=(time.time() - t0) * 1000,
        )
    except Exception as e:
        return AgentResult(
            agent="local_only", query=query,
            external_payload=None, response="", route="local",
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )


# ── Agent 3: External-only ────────────────────────────────────────────────────

async def run_external_only(
    query: str, history: list, external_provider, cache: ResponseCache, model: str = ""
) -> AgentResult:
    t0 = time.time()
    messages = (history or []) + [{"role": "user", "content": query}]
    key = ResponseCache.make_key("external_only", messages, model)

    cached = cache.get(key)
    if cached is not None:
        return AgentResult(
            agent="external_only",
            query=query,
            external_payload=query,
            response=cached,
            route="external",
            from_cache=True,
        )

    try:
        if hasattr(external_provider, "generate_async_messages"):
            response = await external_provider.generate_async_messages(messages) or ""
        else:
            response = await external_provider.generate_async(query) or ""
    except Exception as e:
        return AgentResult(
            agent="external_only", query=query,
            external_payload=query, response="", route="external",
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )

    cache.set(key, response)
    return AgentResult(
        agent="external_only",
        query=query,
        external_payload=query,
        response=response,
        route="external",
        latency_ms=(time.time() - t0) * 1000,
    )


# ── Agent 4: Scan + External (mechanical PII strip baseline) ──────────────────

async def run_scan_external(
    query: str, history: list, external_provider, safety_guard,
    cache: ResponseCache, model: str = ""
) -> AgentResult:
    """
    Mechanical PII strip (SafetyGuard regex) → send sanitized query to external.
    Represents Presidio-style or rule-based DLP baseline.
    """
    t0 = time.time()
    sanitized, pii_found, _, binding_map = safety_guard.scan_and_redact(query)
    pii_types = list({k.strip("[]").split("_")[0] for k in binding_map.keys()})

    messages = (history or []) + [{"role": "user", "content": sanitized}]
    key = ResponseCache.make_key("scan_external", messages, model)

    cached = cache.get(key)
    if cached is not None:
        return AgentResult(
            agent="scan_external",
            query=query,
            external_payload=sanitized,
            response=cached,
            route="external",
            pii_types=pii_types,
            from_cache=True,
        )

    try:
        if hasattr(external_provider, "generate_async_messages"):
            response = await external_provider.generate_async_messages(messages) or ""
        else:
            response = await external_provider.generate_async(sanitized) or ""
    except Exception as e:
        return AgentResult(
            agent="scan_external", query=query,
            external_payload=sanitized, response="", route="external",
            pii_types=pii_types,
            latency_ms=(time.time() - t0) * 1000, error=str(e),
        )

    cache.set(key, response)
    return AgentResult(
        agent="scan_external",
        query=query,
        external_payload=sanitized,
        response=response,
        route="external",
        pii_types=pii_types,
        latency_ms=(time.time() - t0) * 1000,
    )
