"""
Benchmark: Local-only vs External-direct vs Zipsa  (N=100, English)

Dataset   : benchmark_mimic_v1.json — 325 synthetic medical cases
Sampling  : 20 cases per utility_goal category (stratified, seed=42)
Pipeline  : full end-to-end (Stage 1 + 2 + 3) for all three conditions
Metrics   :
  privacy  — % of queries where no PII reached the external provider
  utility  — LLM judge score (1–5) averaged across cases
"""

import asyncio
import json
import random
import sys

sys.path.insert(0, ".")

BENCHMARK_PATH = (
    "/Users/bayes/Library/CloudStorage/"
    "GoogleDrive-sulgik@gmail.com/My Drive/"
    "OpenClawData/workspace/llm-gateway/"
    "06_qa/benchmark_dataset/generated/benchmark_mimic_v1.json"
)

SAMPLE_PER_GOAL = 20   # 5 categories × 20 = 100
BATCH_SIZE      = 5    # concurrent cases (external calls — keep moderate)
RANDOM_SEED     = 42

JUDGE_PROMPT = """\
You are an expert evaluator. Given an original question and an answer, \
rate the answer on a scale from 1 to 5.

ORIGINAL QUESTION:
{question}

ANSWER:
{answer}

Scoring rubric:
5 — Fully correct, directly addresses the question, appropriately detailed
4 — Mostly correct and helpful, minor gaps
3 — Partially addresses the question, notable gaps or inaccuracies
2 — Tangentially relevant but misses the core request
1 — Incorrect, unhelpful, or refuses to answer when it should not

Respond with ONLY a single integer (1–5). No explanation."""


def pii_in_text(text: str, safety) -> bool:
    _, found, _, _ = safety.scan_and_redact(text)
    return found


async def judge(question: str, answer: str, external_provider) -> int:
    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    try:
        raw = await external_provider.generate_async(prompt)
        if raw:
            for tok in raw.strip().split():
                if tok.isdigit() and 1 <= int(tok) <= 5:
                    return int(tok)
    except Exception as e:
        print(f"  [Judge Error] {e}")
    return 0


async def run_local(query: str, ollama) -> str:
    from source.relay.providers.local_provider import LocalProvider
    lp = LocalProvider(ollama_client=ollama)
    return await lp.generate_async_messages([{"role": "user", "content": query}]) or ""


async def run_ext_direct(query: str, ext_provider) -> str:
    try:
        if hasattr(ext_provider, "generate_async_messages"):
            return await ext_provider.generate_async_messages(
                [{"role": "user", "content": query}]
            ) or ""
        return await ext_provider.generate_async(query) or ""
    except Exception as e:
        return f"(error: {e})"


async def run_one(case: dict, safety, orchestrator, ext_provider, ollama) -> dict:
    q = case["prompt_en"]
    print(f"  [{case['id']}] {case['utility_goal'][:40]}")

    # ── Three conditions ──────────────────────────────────────────────────────
    zipsa_result  = await orchestrator.process_request(user_query=q)
    zipsa_answer  = zipsa_result["final_answer"]
    zipsa_ext_q   = zipsa_result["steps"].get("reformulated_query")

    local_answer  = await run_local(q, ollama)
    ext_answer    = await run_ext_direct(q, ext_provider)

    # ── Privacy ───────────────────────────────────────────────────────────────
    zipsa_pii_leaked   = pii_in_text(zipsa_ext_q, safety) if zipsa_ext_q else False
    ext_direct_pii     = pii_in_text(q, safety)

    # ── Utility (LLM judge) ───────────────────────────────────────────────────
    local_score  = await judge(q, local_answer,  ext_provider)
    zipsa_score  = await judge(q, zipsa_answer,  ext_provider)
    ext_score    = await judge(q, ext_answer,    ext_provider)

    route = "hybrid" if zipsa_result["provider"] != "local" else "local"
    print(f"    route={route} | pii_leak={zipsa_pii_leaked} | "
          f"scores: local={local_score} zipsa={zipsa_score} ext={ext_score}")

    return {
        "id":                 case["id"],
        "utility_goal":       case["utility_goal"],
        "route":              route,
        "zipsa_pii_leaked":   zipsa_pii_leaked,
        "ext_direct_pii":     ext_direct_pii,
        "local_utility":      local_score,
        "zipsa_utility":      zipsa_score,
        "ext_utility":        ext_score,
        "zipsa_answer":       zipsa_answer[:200],
        "local_answer":       local_answer[:200],
        "ext_answer":         ext_answer[:200],
    }


async def main():
    from source.relay.safety import SafetyGuard
    from source.relay.config import get_settings
    from source.relay.ollama import OllamaClient
    from source.relay.orchestrator import RelayOrchestrator
    from source.relay.providers.gemini_provider import GeminiProvider

    # ── Load & stratified sample ──────────────────────────────────────────────
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    from collections import defaultdict
    by_goal: dict[str, list] = defaultdict(list)
    for c in dataset:
        by_goal[c["utility_goal"]].append(c)

    rng = random.Random(RANDOM_SEED)
    sample = []
    for goal, cases in sorted(by_goal.items()):
        n = min(SAMPLE_PER_GOAL, len(cases))
        sample.extend(rng.sample(cases, n))
    rng.shuffle(sample)

    print(f"Sampled {len(sample)} cases from {len(by_goal)} goal categories")
    from collections import Counter
    for g, cnt in Counter(c["utility_goal"] for c in sample).most_common():
        print(f"  {cnt:3d}  {g}")
    print()

    # ── Init ──────────────────────────────────────────────────────────────────
    config       = get_settings()
    safety       = SafetyGuard()
    ollama       = OllamaClient(model=config.local_model, host=config.local_host)
    ext_provider = GeminiProvider()
    orchestrator = RelayOrchestrator(config=config)

    # ── Run batches ───────────────────────────────────────────────────────────
    results = []
    for i in range(0, len(sample), BATCH_SIZE):
        batch = sample[i : i + BATCH_SIZE]
        print(f"Batch {i//BATCH_SIZE + 1}/{-(-len(sample)//BATCH_SIZE)}")
        tasks = [run_one(c, safety, orchestrator, ext_provider, ollama) for c in batch]
        batch_out = await asyncio.gather(*tasks, return_exceptions=True)
        for r in batch_out:
            if isinstance(r, Exception):
                print(f"  [Error] {r}")
            else:
                results.append(r)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(results)

    local_privacy  = 1.0
    ext_privacy    = 1 - sum(1 for r in results if r["ext_direct_pii"]) / n
    zipsa_privacy  = 1 - sum(1 for r in results if r["zipsa_pii_leaked"]) / n

    def avg(key):
        scores = [r[key] for r in results if r[key] > 0]
        return round(sum(scores) / len(scores), 2) if scores else 0.0

    local_util = avg("local_utility")
    zipsa_util = avg("zipsa_utility")
    ext_util   = avg("ext_utility")

    print()
    print("=" * 64)
    print(f"RESULTS  (N={n})")
    print("=" * 64)
    print(f"{'Condition':<26} {'Privacy (no PII leak)':>22} {'Utility (1–5)':>14}")
    print("-" * 64)
    print(f"{'local-only':<26} {local_privacy:>21.1%} {local_util:>14.2f}")
    print(f"{'Zipsa':<26} {zipsa_privacy:>21.1%} {zipsa_util:>14.2f}")
    print(f"{'external-direct':<26} {ext_privacy:>21.1%} {ext_util:>14.2f}")
    print("=" * 64)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "summary": {
            "n": n,
            "local_only":      {"privacy_rate": local_privacy,         "utility_avg": local_util},
            "zipsa":           {"privacy_rate": round(zipsa_privacy, 4), "utility_avg": zipsa_util},
            "external_direct": {"privacy_rate": round(ext_privacy, 4),  "utility_avg": ext_util},
        },
        "cases": results,
    }
    with open("benchmark_v2_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("Saved → benchmark_v2_results.json")


if __name__ == "__main__":
    asyncio.run(main())
