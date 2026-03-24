"""OAuth 2.0 flows for Anthropic (Claude) and Google (Gemini)."""

import hashlib
import base64
import secrets
import time
from typing import Optional

import httpx

from .token_store import OAuthToken


# ── PKCE helpers ───────────────────────────────────────────────────────────────

def _generate_pkce() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for PKCE S256."""
    verifier = secrets.token_urlsafe(64)[:128]
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# ── State CSRF store (short-lived, in-memory) ─────────────────────────────────

_pending_states: dict[str, dict] = {}  # state → {provider, verifier, created_at}
_STATE_TTL = 600  # 10 minutes


def create_state(provider: str, verifier: str) -> str:
    _cleanup_expired()
    state = secrets.token_urlsafe(32)
    _pending_states[state] = {
        "provider": provider,
        "verifier": verifier,
        "created_at": time.time(),
    }
    return state


def consume_state(state: str) -> Optional[dict]:
    _cleanup_expired()
    return _pending_states.pop(state, None)


def _cleanup_expired():
    now = time.time()
    expired = [k for k, v in _pending_states.items() if now - v["created_at"] > _STATE_TTL]
    for k in expired:
        del _pending_states[k]


# ── Anthropic OAuth ────────────────────────────────────────────────────────────

class AnthropicOAuth:
    AUTH_URL = "https://console.anthropic.com/oauth/authorize"
    TOKEN_URL = "https://console.anthropic.com/oauth/token"

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_authorization_url(self) -> tuple[str, str]:
        """Return (authorization_url, state)."""
        verifier, challenge = _generate_pkce()
        state = create_state("anthropic", verifier)
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        qs = "&".join(f"{k}={httpx.URL('', params={k: v}).params[k]}" for k, v in params.items())
        return f"{self.AUTH_URL}?{qs}", state

    async def exchange_code(self, code: str, verifier: str) -> OAuthToken:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.TOKEN_URL, data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code_verifier": verifier,
            })
            resp.raise_for_status()
            data = resp.json()
        return OAuthToken(
            provider="anthropic",
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_at=time.time() + data.get("expires_in", 3600),
            scopes=data.get("scope", "").split() if data.get("scope") else [],
        )

    async def refresh_token(self, token: OAuthToken) -> OAuthToken:
        if not token.refresh_token:
            raise ValueError("No refresh token available for Anthropic")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.TOKEN_URL, data={
                "grant_type": "refresh_token",
                "refresh_token": token.refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            })
            resp.raise_for_status()
            data = resp.json()
        return OAuthToken(
            provider="anthropic",
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", token.refresh_token),
            expires_at=time.time() + data.get("expires_in", 3600),
            scopes=data.get("scope", "").split() if data.get("scope") else token.scopes,
        )


# ── Google / Gemini OAuth ──────────────────────────────────────────────────────

class GeminiOAuth:
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    SCOPES = ["https://www.googleapis.com/auth/generative-language"]

    def __init__(self, client_id: str, client_secret: str, redirect_uri: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri

    def get_authorization_url(self) -> tuple[str, str]:
        verifier, challenge = _generate_pkce()
        state = create_state("gemini", verifier)
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.SCOPES),
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
        }
        qs = "&".join(f"{k}={httpx.URL('', params={k: v}).params[k]}" for k, v in params.items())
        return f"{self.AUTH_URL}?{qs}", state

    async def exchange_code(self, code: str, verifier: str) -> OAuthToken:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.TOKEN_URL, data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code_verifier": verifier,
            })
            resp.raise_for_status()
            data = resp.json()
        return OAuthToken(
            provider="gemini",
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token"),
            expires_at=time.time() + data.get("expires_in", 3600),
            scopes=data.get("scope", "").split() if data.get("scope") else self.SCOPES,
        )

    async def refresh_token(self, token: OAuthToken) -> OAuthToken:
        if not token.refresh_token:
            raise ValueError("No refresh token available for Gemini")
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self.TOKEN_URL, data={
                "grant_type": "refresh_token",
                "refresh_token": token.refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            })
            resp.raise_for_status()
            data = resp.json()
        return OAuthToken(
            provider="gemini",
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", token.refresh_token),
            expires_at=time.time() + data.get("expires_in", 3600),
            scopes=data.get("scope", "").split() if data.get("scope") else token.scopes,
        )
