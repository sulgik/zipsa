"""
Large-scale privacy benchmark using benchmark_mimic_v1 (N=325)

Measures only Stage 1 (routing + reformulation) — no external LLM calls.

Privacy metric : PII present in text sent to external provider (0/1 per query → %)
  - local-only      : always 0 (nothing sent externally)
  - external-direct : pii_in_text(prompt_en) for every case
  - zipsa           : pii_in_text(reformulated_query) when hybrid, 0 when local

Routing accuracy : expected label derived from utility_goal
  - hybrid expected : "Medical Consultation / Treatment Recommendation", "Symptom Assessment"
  - local expected  : "Appointment Scheduling", "Prescription Renewal", "Medical Record Access"

Concurrency: BATCH_SIZE cases run in parallel to keep total runtime reasonable.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, ".")

BENCHMARK_PATH = (
    "/Users/bayes/Library/CloudStorage/"
    "GoogleDrive-sulgik@gmail.com/My Drive/"
    "OpenClawData/workspace/llm-gateway/"
    "06_qa/benchmark_dataset/generated/benchmark_mimic_v1.json"
)

BATCH_SIZE = 8  # concurrent cases per batch

# utility_goal → expected route
_HYBRID_GOALS = {
    "Medical Consultation / Treatment Recommendation",
    "Symptom Assessment",
}


def expected_route(utility_goal: str) -> str:
    return "hybrid" if utility_goal in _HYBRID_GOALS else "local"


def pii_in_text(text: str, safety) -> bool:
    _, pii_found, _, _ = safety.scan_and_redact(text)
    return pii_found


async def run_case(case: dict, safety, planner, ollama) -> dict:
    from source.relay.planner import plan_async

    prompt = case["prompt_en"]
    goal = case["utility_goal"]
    exp_route = expected_route(goal)

    _, pii_found, _, binding_map = safety.scan_and_redact(prompt)
    pii_types = list({k.strip("[]").split("_")[0] for k in binding_map.keys()})

    # Stage 1a+b: formal planner (LLM classify + deterministic validate)
    formal_decision = await plan_async(
        query=prompt,
        pii_types=pii_types,
        pii_detected=pii_found,
        ollama_client=ollama,
    )
    actual_route = "hybrid" if formal_decision.decision == "hybrid" else "local"

    ext_query = None
    pii_in_ext = False

    if actual_route == "hybrid":
        ext_query = await ollama.reformulate(text=prompt, conversation_context=None)
        if ext_query:
            pii_in_ext = pii_in_text(ext_query, safety)
        else:
            # reformulation failed → fallback to local
            actual_route = "local"
            pii_in_ext = False

    return {
        "id": case["id"],
        "domain": case["domain"],
        "utility_goal": goal,
        "risk_type": case["risk_type"],
        "expected_route": exp_route,
        "actual_route": actual_route,
        "prompt_pii": pii_found,         # external-direct would expose this
        "zipsa_pii_leaked": pii_in_ext,  # zipsa: PII in outbound text?
        "ext_query_snippet": (ext_query or "")[:120],
    }


async def main():
    from source.relay.safety import SafetyGuard
    from source.relay.config import get_settings
    from source.relay.ollama import OllamaClient
    from source.relay import planner as planner_mod

    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} cases from benchmark_mimic_v1.json")

    config = get_settings()
    safety = SafetyGuard()
    ollama = OllamaClient(model=config.local_model, host=config.local_host)

    results = []
    n = len(dataset)

    for batch_start in range(0, n, BATCH_SIZE):
        batch = dataset[batch_start : batch_start + BATCH_SIZE]
        end_idx = min(batch_start + BATCH_SIZE, n)
        print(f"  [{batch_start+1}–{end_idx}/{n}] running batch...")
        tasks = [run_case(c, safety, planner_mod, ollama) for c in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in batch_results:
            if isinstance(r, Exception):
                print(f"  [Error] {r}")
            else:
                results.append(r)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    n_valid = len(results)

    # Routing accuracy (Zipsa vs expected)
    routing_correct = sum(1 for r in results if r["actual_route"] == r["expected_route"])

    # Privacy rates
    local_privacy   = 1.0   # local-only: never sends anything
    ext_direct_pii  = sum(1 for r in results if r["prompt_pii"])
    ext_privacy     = 1 - ext_direct_pii / n_valid
    zipsa_pii       = sum(1 for r in results if r["zipsa_pii_leaked"])
    zipsa_privacy   = 1 - zipsa_pii / n_valid

    print()
    print("=" * 60)
    print("LARGE-SCALE PRIVACY BENCHMARK  (Stage 1 only, N={})".format(n_valid))
    print("=" * 60)
    print(f"{'Condition':<22} {'Privacy (no PII leak)':>22}")
    print("-" * 46)
    print(f"{'local-only':<22} {local_privacy:>21.1%}")
    print(f"{'zipsa':<22} {zipsa_privacy:>21.1%}")
    print(f"{'external-direct':<22} {ext_privacy:>21.1%}")
    print("-" * 46)
    print(f"Routing accuracy: {routing_correct}/{n_valid} ({routing_correct/n_valid:.1%})")
    print()

    # Per-goal breakdown
    from collections import defaultdict
    by_goal: dict[str, list] = defaultdict(list)
    for r in results:
        by_goal[r["utility_goal"]].append(r)

    print(f"{'Goal':<46} {'N':>4} {'Route OK':>8} {'PII leaked':>10}")
    print("-" * 72)
    for goal, rows in sorted(by_goal.items()):
        ok = sum(1 for r in rows if r["actual_route"] == r["expected_route"])
        leaked = sum(1 for r in rows if r["zipsa_pii_leaked"])
        print(f"{goal:<46} {len(rows):>4} {ok:>8} {leaked:>10}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "summary": {
            "n": n_valid,
            "routing_accuracy": round(routing_correct / n_valid, 3),
            "local_only":      {"privacy_rate": local_privacy},
            "zipsa":           {"privacy_rate": round(zipsa_privacy, 4), "pii_leaked_count": zipsa_pii},
            "external_direct": {"privacy_rate": round(ext_privacy, 4), "pii_count": ext_direct_pii},
        },
        "cases": results,
    }

    out_path = "benchmark_v2_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
