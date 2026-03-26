"""
Zipsa Eval Harness — compare 4 agents on privacy and utility.

Agents: zipsa | local_only | external_only | scan_external
Data:   $EVAL_DATA_DIR/data/sharegpt_global100.jsonl  (or --data-file)
Output: eval/results/eval_{timestamp}.jsonl

Privacy eval:   free, runs on every call (checks external_payload locally)
Utility eval:   uses local LLM judge (free); skip with --privacy-only
External calls: cached in eval/results/response_cache.json

Usage:
    # Privacy only (fast, free — use during development)
    python -m eval.harness --privacy-only

    # Full eval (privacy + utility, external calls cached)
    python -m eval.harness

    # Select specific agents
    python -m eval.harness --agents zipsa,local_only

    # Limit to N turns
    python -m eval.harness --limit 20

Environment:
    EVAL_DATA_DIR   path to llm-gateway/06_qa  (required)
    LOCAL_MODEL     Ollama model for local agent + judge (default: qwen3.5:9b)
    EXTERNAL_PROVIDER  anthropic | openai | gemini (default: anthropic)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running from repo root: python -m eval.harness
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source.relay.safety import SafetyGuard
from source.relay.ollama import OllamaClient
from source.relay.config import get_settings
from source.relay.providers.local_provider import LocalProvider

from eval.cache import ResponseCache
from eval.agents import run_zipsa, run_local_only, run_external_only, run_scan_external
from eval.metrics import privacy_metrics, utility_metrics


# ── Data loading ──────────────────────────────────────────────────────────────

def load_turns(data_file: str, limit: Optional[int] = None):
    """
    Load user turns from sharegpt-format JSONL.
    Yields dicts: {conversation_id, turn, query, history}
    where history is the list of prior turns (for multi-turn context).
    """
    turns = []
    with open(data_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            conv_id = record.get("id", record.get("conversation_id", ""))
            conversation = record.get("conversation", [])

            history = []
            for turn in conversation:
                role = turn.get("role", "")
                text = turn.get("text", "")
                if role == "user":
                    turns.append({
                        "conversation_id": conv_id,
                        "turn": turn.get("turn", len(history)),
                        "query": text,
                        "history": list(history),
                    })
                history.append({"role": role, "content": text})

    if limit:
        turns = turns[:limit]
    return turns


# ── Result record ─────────────────────────────────────────────────────────────

def make_result_record(turn: dict, agent_results: dict, privacy: dict, utility: dict) -> dict:
    return {
        "conversation_id": turn["conversation_id"],
        "turn": turn["turn"],
        "query": turn["query"],
        "agents": {
            name: {
                "external_payload": r.external_payload,
                "response": r.response[:500] if r.response else "",
                "route": r.route,
                "pii_types": r.pii_types,
                "latency_ms": round(r.latency_ms, 1),
                "from_cache": r.from_cache,
                "error": r.error,
            }
            for name, r in agent_results.items()
        },
        "privacy": privacy,
        "utility": utility,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_eval(args):
    # ── Config ──
    data_dir = os.getenv("EVAL_DATA_DIR", "")
    if not data_dir:
        print("ERROR: EVAL_DATA_DIR not set. Point it to llm-gateway/06_qa/")
        sys.exit(1)

    data_file = args.data_file or os.path.join(data_dir, "data", "sharegpt_global100.jsonl")
    if not os.path.exists(data_file):
        print(f"ERROR: Data file not found: {data_file}")
        sys.exit(1)

    selected_agents = set(args.agents.split(",")) if args.agents else {
        "zipsa", "local_only", "external_only", "scan_external"
    }

    print(f"Data:    {data_file}")
    print(f"Agents:  {', '.join(sorted(selected_agents))}")
    print(f"Mode:    {'privacy-only' if args.privacy_only else 'privacy + utility'}")
    print()

    # ── Init components ──
    settings = get_settings()
    cache = ResponseCache()
    safety = SafetyGuard()
    ollama = OllamaClient(model=settings.local_model, host=settings.local_host)
    local_provider = LocalProvider(ollama_client=ollama)

    orchestrator = None
    external_provider = None

    if "zipsa" in selected_agents:
        from source.relay.orchestrator import RelayOrchestrator
        orchestrator = RelayOrchestrator()

    if "external_only" in selected_agents or "scan_external" in selected_agents:
        from source.relay.orchestrator import PROVIDERS
        provider_cls = PROVIDERS.get(settings.external_provider)
        if provider_cls:
            external_provider = provider_cls()
        else:
            print(f"WARNING: Unknown provider {settings.external_provider!r}, skipping external agents")
            selected_agents -= {"external_only", "scan_external"}

    print(f"Cache:   {cache.stats()['entries']} existing entries at {cache.stats()['path']}")

    # ── Load data ──
    turns = load_turns(data_file, limit=args.limit)
    print(f"Turns:   {len(turns)} to evaluate\n")

    # ── Output ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"eval_{timestamp}.jsonl"
    summary_counts = {a: {"turns": 0, "leaked": 0, "external_calls": 0} for a in selected_agents}

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, turn in enumerate(turns):
            print(f"[{i+1:3}/{len(turns)}] conv={turn['conversation_id']} turn={turn['turn']}: {turn['query'][:60]}")

            # Run selected agents
            agent_results = {}
            tasks = []

            async def _run_agent(name):
                if name == "zipsa" and orchestrator:
                    return name, await run_zipsa(turn["query"], turn["history"], orchestrator)
                elif name == "local_only":
                    return name, await run_local_only(turn["query"], turn["history"], local_provider)
                elif name == "external_only" and external_provider:
                    return name, await run_external_only(turn["query"], turn["history"], external_provider, cache, settings.external_provider)
                elif name == "scan_external" and external_provider:
                    return name, await run_scan_external(turn["query"], turn["history"], external_provider, safety, cache, settings.external_provider)

            results = await asyncio.gather(*[_run_agent(name) for name in selected_agents])
            for pair in results:
                if pair:
                    name, result = pair
                    agent_results[name] = result

            # Privacy metrics (free)
            privacy = {
                name: privacy_metrics(result, safety)
                for name, result in agent_results.items()
            }

            # Utility metrics (local LLM judge)
            utility = {}
            if not args.privacy_only:
                util_results = await asyncio.gather(*[
                    utility_metrics(result, ollama)
                    for result in agent_results.values()
                ])
                utility = {
                    name: score
                    for name, score in zip(agent_results.keys(), util_results)
                }

            # Summary tracking
            for name, result in agent_results.items():
                summary_counts[name]["turns"] += 1
                if privacy[name]["leaked"]:
                    summary_counts[name]["leaked"] += 1
                if result.external_payload and not result.from_cache:
                    summary_counts[name]["external_calls"] += 1

            # Write result
            record = make_result_record(turn, agent_results, privacy, utility)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            # Per-turn summary
            for name in selected_agents:
                p = privacy.get(name, {})
                u = utility.get(name, {})
                leak_str = "LEAK" if p.get("leaked") else "ok  "
                util_str = f"util={u['utility_score']:.1f}" if u and u.get("utility_score") else "util=--"
                cache_str = "(cached)" if agent_results.get(name, {}).from_cache if hasattr(agent_results.get(name, {}), 'from_cache') else False else ""
                r = agent_results.get(name)
                cache_str = "(cached)" if r and r.from_cache else ""
                print(f"         {name:<18} {leak_str}  {util_str}  {r.route if r else '?':<8} {cache_str}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Results: {out_path}")
    print(f"\n{'Agent':<18} {'Turns':>6} {'Leaked':>8} {'Leak%':>7} {'ExtCalls':>10}")
    print("-" * 55)
    for name, c in summary_counts.items():
        if c["turns"] == 0:
            continue
        pct = c["leaked"] / c["turns"] * 100
        print(f"{name:<18} {c['turns']:>6} {c['leaked']:>8} {pct:>6.1f}%  {c['external_calls']:>8}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Zipsa agent comparison eval")
    parser.add_argument("--data-file", help="Path to JSONL data file (overrides EVAL_DATA_DIR)")
    parser.add_argument("--agents", help="Comma-separated agents: zipsa,local_only,external_only,scan_external")
    parser.add_argument("--limit", type=int, help="Max number of turns to evaluate")
    parser.add_argument("--privacy-only", action="store_true", help="Skip utility judge (faster, 0 extra LLM calls)")
    args = parser.parse_args()
    asyncio.run(run_eval(args))


# Fix: Optional import at top level
from typing import Optional

if __name__ == "__main__":
    main()
