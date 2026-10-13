"""Persistent local chat history, isolated by username."""

import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Tuple


def _database_path() -> Path:
    configured_path = os.environ.get("PROMPT_AI_HISTORY_DB")
    if configured_path:
        return Path(configured_path).expanduser()
    return Path.home() / ".python_prompt_ai" / "chat_history.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_history_db() -> None:
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES chat_sessions(session_id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def create_session(username: str, title: str = "New chat") -> str:
    session_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_sessions (session_id, username, title, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, username, title, created_at),
        )
    return session_id


def get_user_sessions(username: str) -> List[Tuple[str, str]]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT session_id, title
            FROM chat_sessions AS sessions
            WHERE username = ?
              AND EXISTS (
                  SELECT 1
                  FROM chat_messages AS messages
                  WHERE messages.session_id = sessions.session_id
              )
            ORDER BY created_at DESC
            """,
            (username,),
        ).fetchall()
    return [(str(session_id), str(title)) for session_id, title in rows]


def delete_session(session_id: str, username: str) -> None:
    """Delete one conversation and its messages, verifying ownership."""
    with _connection() as connection:
        owner = connection.execute(
            "SELECT 1 FROM chat_sessions WHERE session_id = ? AND username = ?",
            (session_id, username),
        ).fetchone()
        if owner is None:
            raise LookupError("Chat session was not found for this user.")
        connection.execute(
            "DELETE FROM chat_messages WHERE session_id = ?",
            (session_id,),
        )
        connection.execute(
            "DELETE FROM chat_sessions WHERE session_id = ? AND username = ?",
            (session_id, username),
        )


def delete_user_history(username: str) -> None:
    """Delete all conversations and messages owned by one user."""
    with _connection() as connection:
        connection.execute(
            """
            DELETE FROM chat_messages
            WHERE session_id IN (
                SELECT session_id FROM chat_sessions WHERE username = ?
            )
            """,
            (username,),
        )
        connection.execute(
            "DELETE FROM chat_sessions WHERE username = ?",
            (username,),
        )


def save_message(session_id: str, username: str, role: str, content: str) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        owner = connection.execute(
            "SELECT 1 FROM chat_sessions WHERE session_id = ? AND username = ?",
            (session_id, username),
        ).fetchone()
        if owner is None:
            raise LookupError("Chat session was not found for this user.")
        connection.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, created_at),
        )
        if role == "user":
            connection.execute(
                """
                UPDATE chat_sessions
                SET title = CASE WHEN title = 'New chat' THEN ? ELSE title END
                WHERE session_id = ? AND username = ?
                """,
                (content[:60], session_id, username),
            )


def load_messages(session_id: str, username: str) -> List[Dict[str, str]]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT messages.role, messages.content
            FROM chat_messages AS messages
            JOIN chat_sessions AS sessions
              ON sessions.session_id = messages.session_id
            WHERE messages.session_id = ? AND sessions.username = ?
            ORDER BY messages.message_id
            """,
            (session_id, username),
        ).fetchall()
    return [{"role": str(role), "content": str(content)} for role, content in rows]
