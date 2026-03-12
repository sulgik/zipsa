"""
In-memory session store for dual-history multi-turn conversations.

Each session maintains two parallel histories:
  raw_history       — original messages with PII (local LLM only, never sent externally)
  sanitized_history — reformulated/depersonalized messages (sent to external provider)

Assistant responses are identical in both histories since the local LLM
generates them and they contain no raw PII.
"""
from typing import Dict, List

# session_id -> {"raw_history": [...], "sanitized_history": [...]}
_store: Dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    if session_id not in _store:
        _store[session_id] = {"raw_history": [], "sanitized_history": []}
    return _store[session_id]


def append_turn(
    session_id: str,
    raw_user: str,
    sanitized_user: str,
    assistant: str,
) -> None:
    """Append one completed turn to both histories."""
    s = get_session(session_id)
    s["raw_history"].append({"role": "user",      "content": raw_user})
    s["raw_history"].append({"role": "assistant",  "content": assistant})
    s["sanitized_history"].append({"role": "user",      "content": sanitized_user})
    s["sanitized_history"].append({"role": "assistant",  "content": assistant})


def clear_session(session_id: str) -> None:
    _store.pop(session_id, None)
