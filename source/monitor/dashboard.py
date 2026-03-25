"""
Zipsa Monitoring Dashboard — Read-Only

Reads from logs/relay.jsonl and renders a real-time view of:
  - PII detection events (type, frequency, routing decision)
  - Routing distribution (local vs hybrid)
  - Latency by stage
  - Leakage events (reformulation PII leak caught)

Run with:
  python -m source.monitor.dashboard
  or
  python -m source.monitor.dashboard --port 7861
"""

import json
import os
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install 'zipsa[monitor]' or pip install gradio")
    sys.exit(1)


LOG_PATH = Path(os.environ.get("ZIPSA_LOG_PATH", "logs/relay.jsonl"))
REFRESH_INTERVAL_SEC = 5


# ── Log parsing ───────────────────────────────────────────────────────────────

def load_events(hours: int = 24) -> List[Dict[str, Any]]:
    """Load events from relay.jsonl within the last N hours."""
    if not LOG_PATH.exists():
        return []
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    events = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                ts = datetime.fromisoformat(ev.get("timestamp", ""))
                if ts >= cutoff:
                    events.append(ev)
            except Exception:
                continue
    return events


def compute_stats(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_requests = 0
    routing = Counter()
    pii_types = Counter()
    injection_risks = Counter()
    latencies_local = []
    latencies_external = []
    leakage_events = 0
    fallback_events = 0

    seen_traces = set()

    for ev in events:
        step = ev.get("step_name", "")
        trace = ev.get("trace_id", "")
        flags = ev.get("safety_flags", {})
        output = ev.get("raw_output", {})

        if step == "complete" and trace not in seen_traces:
            seen_traces.add(trace)
            total_requests += 1

        if step == "planning" and isinstance(output, dict):
            path = output.get("execution_path", "")
            if path:
                routing[path] += 1
            for pt in output.get("pii_types", []):
                pii_types[pt.upper()] += 1
            inj = output.get("injection_risk", "")
            if inj:
                injection_risks[inj] += 1

        if step == "local_only":
            lat = ev.get("latency_ms", 0)
            if lat:
                latencies_local.append(lat)

        if step == "external":
            lat = ev.get("latency_ms", 0)
            if lat:
                latencies_external.append(lat)
            if flags.get("blocked"):
                fallback_events += 1

        if step == "planning" and isinstance(output, dict):
            if "leaked_binding" in str(output) or "Reformulation leaked" in str(output):
                leakage_events += 1

    avg_local  = sum(latencies_local)  / len(latencies_local)  if latencies_local  else 0
    avg_ext    = sum(latencies_external) / len(latencies_external) if latencies_external else 0

    return {
        "total_requests": total_requests,
        "routing": dict(routing),
        "pii_types": dict(pii_types.most_common(10)),
        "injection_risks": dict(injection_risks),
        "avg_local_ms": round(avg_local, 1),
        "avg_external_ms": round(avg_ext, 1),
        "leakage_caught": leakage_events,
        "fallbacks": fallback_events,
    }


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def render_dashboard(hours: int = 24):
    events = load_events(hours=hours)
    stats = compute_stats(events)

    # Summary cards
    summary = f"""
## 📊 Zipsa Privacy Monitor — Last {hours}h

| Metric | Value |
|--------|-------|
| Total Requests | **{stats['total_requests']}** |
| Local-only | **{stats['routing'].get('local', 0)}** |
| Hybrid (local + external) | **{stats['routing'].get('hybrid', 0)}** |
| PII Leakage Caught | **{stats['leakage_caught']}** 🛡️ |
| External Fallbacks | **{stats['fallbacks']}** |
| Avg Local Latency | **{stats['avg_local_ms']} ms** |
| Avg External Latency | **{stats['avg_external_ms']} ms** |
"""

    # PII breakdown
    if stats["pii_types"]:
        pii_rows = "\n".join(
            f"| {k} | {v} |" for k, v in sorted(stats["pii_types"].items(), key=lambda x: -x[1])
        )
        pii_table = f"\n## 🔍 PII Type Breakdown\n\n| Type | Count |\n|------|-------|\n{pii_rows}"
    else:
        pii_table = "\n## 🔍 PII Type Breakdown\n\n_No PII detected yet._"

    # Injection risk
    if stats["injection_risks"]:
        risk_rows = "\n".join(
            f"| {k} | {v} |" for k, v in sorted(stats["injection_risks"].items(), key=lambda x: -x[1])
        )
        risk_table = f"\n## ⚠️ Injection Risk Events\n\n| Risk Level | Count |\n|------------|-------|\n{risk_rows}"
    else:
        risk_table = "\n## ⚠️ Injection Risk Events\n\n_None detected._"

    updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    footer = f"\n\n---\n_Last updated: {updated} · Log: {LOG_PATH}_"

    return summary + pii_table + risk_table + footer


def build_ui():
    with gr.Blocks(title="Zipsa Privacy Monitor", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🛡️ Zipsa Privacy Monitor")

        with gr.Row():
            hours_slider = gr.Slider(1, 168, value=24, step=1, label="Time window (hours)")
            refresh_btn  = gr.Button("🔄 Refresh", variant="primary")

        output_md = gr.Markdown(render_dashboard())

        def update(hours):
            return render_dashboard(int(hours))

        refresh_btn.click(fn=update, inputs=hours_slider, outputs=output_md)
        hours_slider.change(fn=update, inputs=hours_slider, outputs=output_md)

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    ui = build_ui()
    print(f"🛡️  Zipsa Monitor running at http://{args.host}:{args.port}")
    ui.launch(server_name=args.host, server_port=args.port, share=False)
