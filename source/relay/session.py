"""
SQLite-backed session store for dual-thread multi-turn conversations.

Each session maintains two linked conversation threads:
  main_thread — the full local conversation with original user turns and final answers
  sub_thread  — the external-safe conversation used only for hybrid knowledge access

The sub-thread is intentionally sparse: local-only turns do not appear there.

Sessions expire after SESSION_TTL_DAYS days of inactivity (default: 7).
"""

import json
import sqlite3
import time
import os
from pathlib import Path
from typing import Dict

SESSION_TTL_DAYS = 7
SESSION_TTL_SECONDS = SESSION_TTL_DAYS * 86400

# Default DB path — can override via ZIPSA_SESSION_DB env var
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "sessions.db"


def _get_db_path() -> Path:
    path = Path(os.environ.get("ZIPSA_SESSION_DB", DEFAULT_DB_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   TEXT PRIMARY KEY,
            main_thread  TEXT NOT NULL DEFAULT '[]',
            sub_thread   TEXT NOT NULL DEFAULT '[]',
            updated_at   REAL NOT NULL
        )
    """)
    conn.commit()


def get_session(session_id: str) -> dict:
    """Load session from SQLite. Creates a new empty session if not found."""
    with _get_conn() as conn:
        _init_db(conn)
        row = conn.execute(
            "SELECT main_thread, sub_thread FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row:
            return {
                "main_thread": json.loads(row["main_thread"]),
                "sub_thread":  json.loads(row["sub_thread"]),
            }
        # New session — insert and return empty
        conn.execute(
            "INSERT INTO sessions (session_id, main_thread, sub_thread, updated_at) VALUES (?, '[]', '[]', ?)",
            (session_id, time.time())
        )
        conn.commit()
        return {"main_thread": [], "sub_thread": []}


def _save_session(conn: sqlite3.Connection, session_id: str, session: dict) -> None:
    conn.execute(
        """INSERT INTO sessions (session_id, main_thread, sub_thread, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               main_thread = excluded.main_thread,
               sub_thread  = excluded.sub_thread,
               updated_at  = excluded.updated_at""",
        (
            session_id,
            json.dumps(session["main_thread"], ensure_ascii=False),
            json.dumps(session["sub_thread"],  ensure_ascii=False),
            time.time(),
        )
    )
    conn.commit()


def append_main_turn(session_id: str, user: str, assistant: str) -> None:
    """Append one completed turn to the local main thread."""
    with _get_conn() as conn:
        _init_db(conn)
        session = get_session(session_id)
        session["main_thread"].append({"role": "user",      "content": user})
        session["main_thread"].append({"role": "assistant", "content": assistant})
        _save_session(conn, session_id, session)


def append_sub_turn(session_id: str, user: str, assistant: str) -> None:
    """Append one completed turn to the external-safe sub-thread."""
    with _get_conn() as conn:
        _init_db(conn)
        session = get_session(session_id)
        session["sub_thread"].append({"role": "user",      "content": user})
        session["sub_thread"].append({"role": "assistant", "content": assistant})
        _save_session(conn, session_id, session)


def clear_session(session_id: str) -> None:
    """Delete a session from the store."""
    with _get_conn() as conn:
        _init_db(conn)
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()


def purge_expired_sessions() -> int:
    """Delete sessions not updated within SESSION_TTL_DAYS. Returns count deleted."""
    cutoff = time.time() - SESSION_TTL_SECONDS
    with _get_conn() as conn:
        _init_db(conn)
        cursor = conn.execute(
            "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)
        )
        conn.commit()
        return cursor.rowcount
