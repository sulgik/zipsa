"""In-memory OAuth token storage with optional encrypted file persistence."""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class OAuthToken:
    provider: str           # "anthropic" | "gemini"
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: float = 0.0  # unix timestamp
    scopes: list[str] = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = []

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return False  # no expiry info → assume valid
        return time.time() >= (self.expires_at - 60)  # 60s buffer


class TokenStore:
    """Thread-safe in-memory token store with optional Fernet-encrypted persistence."""

    def __init__(self, persist_path: Optional[Path] = None, encryption_key: str = ""):
        self._tokens: dict[str, OAuthToken] = {}
        self._persist_path = persist_path
        self._fernet = None
        if encryption_key:
            from cryptography.fernet import Fernet
            self._fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        if persist_path and persist_path.exists():
            self._load()

    def get(self, provider: str) -> Optional[OAuthToken]:
        return self._tokens.get(provider)

    def set(self, provider: str, token: OAuthToken) -> None:
        self._tokens[provider] = token
        self._save()

    def delete(self, provider: str) -> None:
        self._tokens.pop(provider, None)
        self._save()

    def _save(self):
        if not self._persist_path:
            return
        data = json.dumps({k: asdict(v) for k, v in self._tokens.items()}).encode()
        if self._fernet:
            data = self._fernet.encrypt(data)
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_bytes(data)

    def _load(self):
        try:
            data = self._persist_path.read_bytes()
            if self._fernet:
                data = self._fernet.decrypt(data)
            raw = json.loads(data)
            for k, v in raw.items():
                self._tokens[k] = OAuthToken(**v)
        except Exception as e:
            print(f"[TokenStore] Failed to load persisted tokens: {e}")


# ── Singleton ──────────────────────────────────────────────────────────────────
_store: Optional[TokenStore] = None


def get_token_store() -> TokenStore:
    global _store
    if _store is None:
        import os
        encryption_key = os.getenv("TOKEN_ENCRYPTION_KEY", "")
        persist_path = None
        if encryption_key:
            persist_path = Path(__file__).resolve().parent.parent.parent / ".tokens.enc"
        _store = TokenStore(persist_path=persist_path, encryption_key=encryption_key)
    return _store
