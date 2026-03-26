"""
Response cache — avoids redundant external LLM calls across eval runs.

Cache key: SHA-256 of (agent_name, serialized_messages, model_id)
Cache file: eval/results/response_cache.json  (or EVAL_CACHE_PATH env var)

Usage:
    cache = ResponseCache()
    resp = cache.get(key)
    if resp is None:
        resp = await call_external(...)
        cache.set(key, resp)
"""

import hashlib
import json
import os
from typing import Optional


class ResponseCache:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv(
            "EVAL_CACHE_PATH",
            os.path.join(os.path.dirname(__file__), "results", "response_cache.json"),
        )
        self._data: dict = {}
        self._dirty = False
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str):
        self._data[key] = value
        self._dirty = True
        self._flush()

    def _flush(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def stats(self) -> dict:
        return {"entries": len(self._data), "path": self.path}

    @staticmethod
    def make_key(agent: str, messages: list, model: str = "") -> str:
        payload = json.dumps(
            {"agent": agent, "messages": messages, "model": model},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
