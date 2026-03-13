"""
In-memory session store for dual-thread multi-turn conversations.

Each session maintains two linked conversation threads:
  main_thread — the full local conversation with original user turns and final answers
  sub_thread  — the external-safe conversation used only for hybrid knowledge access

The sub-thread is intentionally sparse: local-only turns do not have to appear there.
"""
from typing import Dict

# session_id -> {"main_thread": [...], "sub_thread": [...]}
_store: Dict[str, dict] = {}


def get_session(session_id: str) -> dict:
    if session_id not in _store:
        _store[session_id] = {"main_thread": [], "sub_thread": []}
    return _store[session_id]


def append_main_turn(
    session_id: str,
    user: str,
    assistant: str,
) -> None:
    """Append one completed turn to the local main thread only."""
    s = get_session(session_id)
    s["main_thread"].append({"role": "user", "content": user})
    s["main_thread"].append({"role": "assistant", "content": assistant})


def append_sub_turn(
    session_id: str,
    user: str,
    assistant: str,
) -> None:
    """Append one completed turn to the external-safe sub-thread only."""
    s = get_session(session_id)
    s["sub_thread"].append({"role": "user", "content": user})
    s["sub_thread"].append({"role": "assistant", "content": assistant})


def clear_session(session_id: str) -> None:
    _store.pop(session_id, None)
