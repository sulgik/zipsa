"""Zipsa: privacy-preserving hybrid AI gateway.

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from source.relay.orchestrator import RelayOrchestrator
from source.relay.config import get_settings
from source.adapters.openai_compat import router as openai_router
from source.auth.router import router as auth_router

settings = get_settings()

app = FastAPI(
    title="Zipsa",
    version="0.4.0",
    description="Privacy-preserving hybrid AI gateway",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = RelayOrchestrator()
security = HTTPBearer(auto_error=False)

app.state.orchestrator = orchestrator
app.include_router(openai_router)
app.include_router(auth_router)


class RelayRequest(BaseModel):
    user_query: str
    session_id: Optional[str] = None


class RelayResponse(BaseModel):
    final_answer: str
    steps: dict
    provider: str
    timings: dict
    latency_ms: float
    trace_id: str
    pii_detected: bool = False
    pii_types: list = []
    trace: list[str] = []


@app.post("/relay", response_model=RelayResponse)
async def relay_endpoint(
    request: RelayRequest,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    if not settings.demo_mode and (creds is None or not creds.credentials):
        raise HTTPException(status_code=401, detail="Bearer token required")

    if not request.user_query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    api_key = creds.credentials if creds else None
    result = await orchestrator.process_request(
        user_query=request.user_query,
        api_key=api_key,
        session_id=request.session_id,
    )
    return result


@app.get("/")
async def chat_ui():
    return FileResponse("source/ui/chat.html")


@app.get("/health")
async def health_check():
    status = {
        "status": "ok",
        "local_model": settings.local_model,
        "external_provider": settings.external_provider,
        "demo_mode": settings.demo_mode,
    }
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.local_host}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                status["ollama"] = "connected"
                status["models"] = models
            else:
                status["ollama"] = "error"
    except Exception:
        status["ollama"] = "unreachable"
    return status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
