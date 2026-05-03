"""SQLite3 database layer for NPS SS3861 tutor.

Stores user identities, conversation sessions, and message history.
No evaluation scores or tool usage are persisted here.

Phase 2 checkpoint — run unit tests:
    python db/database.py
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime


# ── Schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL CHECK(role IN ('student', 'instructor')),
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    started_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  INTEGER NOT NULL REFERENCES conversations(id),
    role             TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def _connect(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Init ──────────────────────────────────────────────────────────────────────

def init_db(db_path: str) -> None:
    """Create all tables if they do not exist."""
    with _connect(db_path) as conn:
        conn.executescript(_DDL)


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(db_path: str, name: str, role: str) -> int:
    """Insert a new user and return their id."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO users (name, role) VALUES (?, ?)", (name, role)
        )
        return cur.lastrowid


def get_user(db_path: str, user_id: int) -> dict | None:
    """Return a user dict or None if not found."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, name, role, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def get_or_create_user(db_path: str, name: str, role: str) -> dict:
    """Return existing user by name, or create and return a new one.

    If multiple users share the same name, returns the most recently created.
    Intended for the Streamlit login flow where no password exists.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, name, role, created_at FROM users "
            "WHERE name = ? ORDER BY id DESC LIMIT 1",
            (name,),
        ).fetchone()
        if row:
            return dict(row)
        cur = conn.execute(
            "INSERT INTO users (name, role) VALUES (?, ?)", (name, role)
        )
        user_id = cur.lastrowid
        row = conn.execute(
            "SELECT id, name, role, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row)


# ── Conversations ─────────────────────────────────────────────────────────────

def start_conversation(db_path: str, user_id: int) -> int:
    """Create a new conversation for user_id and return the conversation id."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO conversations (user_id) VALUES (?)", (user_id,)
        )
        return cur.lastrowid


def get_user_conversations(db_path: str, user_id: int) -> list[dict]:
    """Return all conversations for a user, newest first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, user_id, started_at FROM conversations "
            "WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_conversation(db_path: str, conversation_id: int) -> None:
    """Delete a conversation and all its messages."""
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


# ── Messages ──────────────────────────────────────────────────────────────────

def save_message(db_path: str, conversation_id: int, role: str, content: str) -> int:
    """Persist a single message and return its id."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
            (conversation_id, role, content),
        )
        return cur.lastrowid


def get_conversation_history(
    db_path: str, conversation_id: int, limit: int = 20
) -> list[dict]:
    """Return the last `limit` messages as a list of {role, content} dicts."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
    # Return in chronological order
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


# ── CLI unit tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name

    try:
        init_db(db)
        print("init_db        OK")

        uid = create_user(db, "Alice", "student")
        assert uid == 1
        print("create_user    OK")

        user = get_user(db, uid)
        assert user["name"] == "Alice" and user["role"] == "student"
        print("get_user       OK")

        user2 = get_or_create_user(db, "Alice", "student")
        assert user2["id"] == uid, "should return existing user"
        user3 = get_or_create_user(db, "Bob", "instructor")
        assert user3["name"] == "Bob"
        print("get_or_create  OK")

        cid = start_conversation(db, uid)
        assert cid == 1
        print("start_conversation OK")

        convs = get_user_conversations(db, uid)
        assert len(convs) == 1 and convs[0]["id"] == cid
        print("get_user_conversations OK")

        mid1 = save_message(db, cid, "user", "What is a C&DH system?")
        mid2 = save_message(db, cid, "assistant", "C&DH stands for Command and Data Handling...")
        assert mid1 == 1 and mid2 == 2
        print("save_message   OK")

        history = get_conversation_history(db, cid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        print("get_conversation_history OK")

        # Limit test
        for i in range(25):
            save_message(db, cid, "user", f"msg {i}")
        history_limited = get_conversation_history(db, cid, limit=20)
        assert len(history_limited) == 20
        print("limit           OK")

        print("\nAll tests passed.")
    finally:
        os.unlink(db)
