"""
OpenAI-compatible adapter for OpenClaw (and any OpenAI-compatible client).

Mounts onto an existing FastAPI app via include_router().
"""
import json
import time
import uuid
from collections import deque

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from source.relay.orchestrator import RelayOrchestrator

router = APIRouter()
security = HTTPBearer(auto_error=False)

# In-memory routing log (last 50 requests)
_log: deque[dict] = deque(maxlen=50)


def _extract_latest_user_message(messages: list[dict]) -> str:
    """Return only the latest user message (current turn, raw)."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return (content or "").strip()
    return ""


def _make_footer(result: dict) -> str:
    provider = result.get("provider", "?")
    routing = result.get("routing") or {}
    route_label = routing.get("routed_to", provider)
    is_external = route_label not in ("local",)
    sanitized_symbol = " 🛡️sanitized" if is_external else ""
    return f"\n\n---\n🔒 Zipsa: {route_label}{sanitized_symbol}"


def _sse_stream(final_answer: str, model_id: str) -> StreamingResponse:
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def generate():
        for delta in [
            {"role": "assistant", "content": ""},
            {"content": final_answer},
        ]:
            chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        chunk = {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": len(final_answer.split()), "total_tokens": 0},
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/v1/chat/completions")
async def openai_chat_completions(
    raw: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    body = await raw.json()
    user_query = _extract_latest_user_message(body.get("messages", []))
    if not user_query.strip():
        raise HTTPException(status_code=400, detail="No message content")

    # session_id is an optional custom field for multi-turn context tracking
    session_id = body.get("session_id") or None

    orchestrator: RelayOrchestrator = raw.app.state.orchestrator
    api_key = creds.credentials if creds else None
    result = await orchestrator.process_request(
        user_query=user_query, api_key=api_key, session_id=session_id
    )

    final_answer = (result.get("final_answer") or "") + _make_footer(result)

    _log.appendleft({
        "time": time.strftime("%H:%M:%S"),
        "query": user_query[:80],
        "provider": result.get("provider", "?"),
        "routing": result.get("routing", {}),
        "sensitivity": result.get("steps", {}).get("sensitivity", "?"),
        "latency_ms": result.get("latency_ms", 0),
        "answer_len": len(final_answer),
    })
    print(f"[Zipsa] → {_log[0]['provider']} | PII={_log[0]['sensitivity']} | {_log[0]['answer_len']}chars")

    return _sse_stream(final_answer, body.get("model") or "zipsa")


@router.get("/logs")
async def get_logs():
    return {"count": len(_log), "recent": list(_log)}
