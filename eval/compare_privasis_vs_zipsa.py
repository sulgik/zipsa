"""
Privasis+Ext  vs  Zipsa  — 직접 비교 eval
==========================================
Privasis vanilla_test.parquet에서 도메인별 샘플을 뽑아
두 파이프라인을 동일 입력으로 평가합니다.

평가 지표:
  Privacy  : external_payload에 원본 PII가 얼마나 남는지 (SafetyGuard 기준)
  Utility  : 답변 길이·거부 여부 (LLM judge 없이 휴리스틱, 빠름)
  Latency  : 벽시계 기준

Usage:
    cd /Users/bayes/OpenClawData/workspace/zipsa
    python -m eval.compare_privasis_vs_zipsa
"""

import asyncio, json, sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# .env 먼저 로드 (ANTHROPIC_API_KEY 등이 provider 초기화 전에 필요)
from source.relay.config import _load_dotenv
_load_dotenv()

import pandas as pd

from source.relay.orchestrator import RelayOrchestrator
from source.relay.safety import SafetyGuard
from source.relay.ollama import OllamaClient
from source.relay.config import get_settings
from eval.cache import ResponseCache
from eval.agents import run_zipsa, run_privasis_external

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH  = Path(__file__).parent.parent.parent / "llm-gateway/06_qa/data/privasis-zero/vanilla_test.parquet"
N_PER_TAG  = 3        # 도메인당 샘플 수
TAGS       = ["medical", "legal", "finance", "hr", "comms"]  # 5개 도메인
TOTAL      = N_PER_TAG * len(TAGS)   # 15건
EVAL_MODEL = "qwen3.5:9b"   # eval 전용 로컬 모델 (빠름)

# ── Privacy scorer ────────────────────────────────────────────────────────────

def pii_leak_score(external_payload: str | None, safety: SafetyGuard, original: str) -> dict:
    """외부로 나간 텍스트에서 PII가 얼마나 남았는지 측정."""
    if external_payload is None:
        return {"leaked": False, "pii_count": 0, "pii_types": [], "severity": 0, "not_sent": True}

    _, orig_found, _, orig_map   = safety.scan_and_redact(original)
    _, leak_found, _, leak_map   = safety.scan_and_redact(external_payload)

    orig_types = {k.strip("[]").split("_")[0] for k in orig_map}
    leak_types = {k.strip("[]").split("_")[0] for k in leak_map}

    from source.relay.router import PII_SEVERITY_SCORES
    severity = sum(PII_SEVERITY_SCORES.get(t.upper(), 3) for t in leak_types)

    return {
        "leaked":    leak_found,
        "pii_count": len(leak_map),
        "pii_types": sorted(leak_types),
        "severity":  severity,
        "not_sent":  False,
        "orig_pii_count": len(orig_map),
        # 원본 PII 중 외부로 나간 비율
        "leak_rate": len(leak_map) / max(len(orig_map), 1),
    }


def utility_heuristic(response: str) -> dict:
    if not response or len(response.strip()) < 20:
        return {"ok": False, "reason": "empty", "length": len(response)}
    refusals = ["i cannot", "i can't", "i'm unable", "cannot assist",
                "제공할 수 없", "도움이 되지 않"]
    refused = any(r in response.lower() for r in refusals)
    return {"ok": not refused, "reason": "refused" if refused else "ok", "length": len(response)}


# ── Main ─────────────────────────────────────────────────────────────────────

async def run():
    print("=" * 64)
    print("  Privasis+Ext  vs  Zipsa  —  비교 eval")
    print("=" * 64)

    # ── 데이터 로드 및 샘플링
    df = pd.read_parquet(DATA_PATH)
    samples = []
    for tag in TAGS:
        subset = df[df['record_tags'].str.contains(tag, na=False)]
        picked = subset.sample(min(N_PER_TAG, len(subset)), random_state=42)
        for _, row in picked.iterrows():
            samples.append({
                "tag":         tag,
                "record_type": row["record_type"][:60],
                "query":       row["original_record"],
                "instruction": row["smoothed_instruction"],   # Privasis 지시문
                "sanitized_gt": row["sanitized_record"],      # 정답 sanitized
            })
    print(f"  샘플: {len(samples)}건  ({N_PER_TAG}/도메인 × {len(TAGS)}도메인)\n")

    # ── 컴포넌트 초기화
    settings    = get_settings()
    safety      = SafetyGuard()
    cache       = ResponseCache()
    # eval 전용 올라마: 빠른 모델 사용 (Privasis sanitize + Zipsa local 추론용)
    ollama      = OllamaClient(model=EVAL_MODEL, host=settings.local_host)
    # Zipsa orchestrator — eval 속도를 위해 9b 모델 사용
    eval_config = get_settings()
    eval_config.local_model = EVAL_MODEL
    orchestrator = RelayOrchestrator(config=eval_config)

    from source.relay.orchestrator import PROVIDERS
    provider_cls = PROVIDERS.get(settings.external_provider)
    external_provider = provider_cls() if provider_cls else None
    if external_provider is None:
        print("ERROR: external provider not available"); return

    # ── 결과 수집
    results = []
    agg = {
        "privasis_external": defaultdict(list),
        "zipsa":             defaultdict(list),
    }

    for i, s in enumerate(samples):
        query = s["query"]
        tag   = s["tag"]
        print(f"[{i+1:2}/{len(samples)}] {tag:<8}  {s['record_type'][:50]}")

        # 두 에이전트 병렬 실행
        (priv_res, zipsa_res) = await asyncio.gather(
            run_privasis_external(
                query=query, history=[],
                external_provider=external_provider,
                ollama_client=ollama,
                cache=cache, model=settings.external_provider,
                instruction=s["instruction"],
            ),
            run_zipsa(query=query, history=[], orchestrator=orchestrator),
        )

        priv_priv  = pii_leak_score(priv_res.external_payload,  safety, query)
        zipsa_priv = pii_leak_score(zipsa_res.external_payload, safety, query)
        priv_util  = utility_heuristic(priv_res.response)
        zipsa_util = utility_heuristic(zipsa_res.response)

        for name, res, priv, util in [
            ("privasis_external", priv_res,  priv_priv,  priv_util),
            ("zipsa",             zipsa_res, zipsa_priv, zipsa_util),
        ]:
            agg[name]["leaked"].append(priv["leaked"])
            agg[name]["pii_count"].append(priv["pii_count"])
            agg[name]["severity"].append(priv["severity"])
            agg[name]["not_sent"].append(priv.get("not_sent", False))
            agg[name]["leak_rate"].append(priv.get("leak_rate", 0))
            agg[name]["util_ok"].append(util["ok"])
            agg[name]["latency"].append(res.latency_ms)

        # 한 줄 요약
        def fmt(priv, util, res):
            leak = "LEAK" if priv["leaked"] else ("✉  " if not priv.get("not_sent") else "🔒 ")
            pii  = f"pii={priv['pii_count']}"
            ut   = "✅" if util["ok"] else "❌"
            lat  = f"{res.latency_ms:.0f}ms"
            route = getattr(res, 'route', '?')
            return f"{leak}  {pii}  util={ut}  {route:<8} {lat}"

        print(f"  privasis+ext : {fmt(priv_priv,  priv_util,  priv_res)}")
        print(f"  zipsa        : {fmt(zipsa_priv, zipsa_util, zipsa_res)}")

        results.append({
            "tag": tag, "record_type": s["record_type"],
            "privasis_external": {
                "privacy": priv_priv, "utility": priv_util,
                "latency_ms": round(priv_res.latency_ms, 1),
                "route": priv_res.route, "error": priv_res.error,
                "payload_preview": (priv_res.external_payload or "")[:200],
            },
            "zipsa": {
                "privacy": zipsa_priv, "utility": zipsa_util,
                "latency_ms": round(zipsa_res.latency_ms, 1),
                "route": zipsa_res.route, "error": zipsa_res.error,
                "payload_preview": (zipsa_res.external_payload or "")[:200],
            },
        })

    # ── 집계 요약
    print()
    print("=" * 64)
    print("  결과 요약")
    print("=" * 64)

    def avg(lst): return sum(lst) / max(len(lst), 1)
    def pct(lst): return avg(lst) * 100

    header = f"{'지표':<28} {'Privasis+Ext':>14} {'Zipsa':>10}"
    print(f"\n{header}")
    print("-" * 54)

    pa = agg["privasis_external"]
    za = agg["zipsa"]

    rows = [
        ("외부 전송 안 함 (local-only) %", pct(pa["not_sent"]),   pct(za["not_sent"])),
        ("PII 유출 건수 %",               pct(pa["leaked"]),      pct(za["leaked"])),
        ("평균 외부 노출 PII 수",          avg(pa["pii_count"]),   avg(za["pii_count"])),
        ("평균 PII 심각도 점수",           avg(pa["severity"]),    avg(za["severity"])),
        ("평균 leak_rate (원본 대비)",     avg(pa["leak_rate"]),   avg(za["leak_rate"])),
        ("Utility 성공 %",                pct(pa["util_ok"]),     pct(za["util_ok"])),
        ("평균 latency (ms)",             avg(pa["latency"]),     avg(za["latency"])),
    ]

    for label, pv, zv in rows:
        # privacy 지표는 낮을수록 좋음, utility/local %는 높을수록 좋음
        if "유출" in label or "노출" in label or "심각도" in label or "leak_rate" in label:
            winner = "◀ Zipsa" if zv < pv else ("◀ Priv" if pv < zv else "tie")
        elif "latency" in label:
            winner = "◀ Zipsa" if zv < pv else ("◀ Priv" if pv < zv else "tie")
        else:
            winner = "◀ Zipsa" if zv > pv else ("◀ Priv" if pv > zv else "tie")
        print(f"  {label:<26} {pv:>12.1f}   {zv:>8.1f}   {winner}")

    # 도메인별 breakdown
    print()
    print("도메인별 PII 유출 %")
    print(f"  {'도메인':<10} {'Privasis+Ext':>14} {'Zipsa':>10}")
    print("  " + "-" * 36)
    domain_stats = defaultdict(lambda: {"p_leaked": [], "z_leaked": []})
    for r in results:
        t = r["tag"]
        domain_stats[t]["p_leaked"].append(r["privasis_external"]["privacy"]["leaked"])
        domain_stats[t]["z_leaked"].append(r["zipsa"]["privacy"]["leaked"])
    for tag in TAGS:
        ds = domain_stats[tag]
        pp = pct(ds["p_leaked"])
        zp = pct(ds["z_leaked"])
        print(f"  {tag:<10} {pp:>12.0f}%   {zp:>8.0f}%")

    # 저장
    out_path = Path(__file__).parent / "results" / f"compare_privasis_zipsa_{int(time.time())}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": rows, "results": results}, f, ensure_ascii=False, indent=2,
                  default=lambda o: str(o) if not isinstance(o, (int, float, str, bool, list, dict, type(None))) else o)
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    asyncio.run(run())
