"""FastAPI router for OAuth endpoints (Claude & Gemini)."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse

from source.relay.config import get_settings
from .token_store import get_token_store
from .oauth_flows import AnthropicOAuth, GeminiOAuth, consume_state

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _anthropic_oauth() -> AnthropicOAuth:
    s = get_settings()
    if not s.anthropic_oauth_client_id:
        raise HTTPException(400, "ANTHROPIC_OAUTH_CLIENT_ID not configured")
    return AnthropicOAuth(
        client_id=s.anthropic_oauth_client_id,
        client_secret=s.anthropic_oauth_client_secret,
        redirect_uri=f"{s.oauth_redirect_base}/auth/claude/callback",
    )


def _gemini_oauth() -> GeminiOAuth:
    s = get_settings()
    if not s.google_oauth_client_id:
        raise HTTPException(400, "GOOGLE_OAUTH_CLIENT_ID not configured")
    return GeminiOAuth(
        client_id=s.google_oauth_client_id,
        client_secret=s.google_oauth_client_secret,
        redirect_uri=f"{s.oauth_redirect_base}/auth/gemini/callback",
    )


_SUCCESS_HTML = """\
<!DOCTYPE html>
<html><body style="font-family:sans-serif;text-align:center;padding:4rem">
<h2>&#10003; {provider} OAuth connected</h2>
<p>You can close this window.</p>
</body></html>
"""


# ── Claude / Anthropic ─────────────────────────────────────────────────────────

@router.get("/claude")
async def claude_authorize():
    """Redirect user to Anthropic OAuth consent screen."""
    url, _ = _anthropic_oauth().get_authorization_url()
    return RedirectResponse(url)


@router.get("/claude/callback")
async def claude_callback(code: str, state: str):
    """Handle Anthropic OAuth callback."""
    pending = consume_state(state)
    if not pending or pending["provider"] != "anthropic":
        raise HTTPException(400, "Invalid or expired OAuth state")

    oauth = _anthropic_oauth()
    token = await oauth.exchange_code(code, pending["verifier"])
    get_token_store().set("anthropic", token)
    return HTMLResponse(_SUCCESS_HTML.format(provider="Claude"))


@router.get("/claude/status")
async def claude_status():
    token = get_token_store().get("anthropic")
    if not token:
        return {"connected": False}
    return {
        "connected": True,
        "expired": token.is_expired,
        "expires_at": token.expires_at,
        "scopes": token.scopes,
    }


@router.post("/claude/revoke")
async def claude_revoke():
    get_token_store().delete("anthropic")
    return {"revoked": True}


# ── Gemini / Google ────────────────────────────────────────────────────────────

@router.get("/gemini")
async def gemini_authorize():
    """Redirect user to Google OAuth consent screen."""
    url, _ = _gemini_oauth().get_authorization_url()
    return RedirectResponse(url)


@router.get("/gemini/callback")
async def gemini_callback(code: str, state: str):
    """Handle Google OAuth callback."""
    pending = consume_state(state)
    if not pending or pending["provider"] != "gemini":
        raise HTTPException(400, "Invalid or expired OAuth state")

    oauth = _gemini_oauth()
    token = await oauth.exchange_code(code, pending["verifier"])
    get_token_store().set("gemini", token)
    return HTMLResponse(_SUCCESS_HTML.format(provider="Gemini"))


@router.get("/gemini/status")
async def gemini_status():
    token = get_token_store().get("gemini")
    if not token:
        return {"connected": False}
    return {
        "connected": True,
        "expired": token.is_expired,
        "expires_at": token.expires_at,
        "scopes": token.scopes,
    }


@router.post("/gemini/revoke")
async def gemini_revoke():
    get_token_store().delete("gemini")
    return {"revoked": True}
