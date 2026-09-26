import sqlite3
import json
from datetime import datetime

DB_PATH = "storage_history.db"

def init_history_db():
    """Initializes a local history tracker on the server filesystem."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # Table for storing unique user-managed chat sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT,
                session_title TEXT,
                created_at TEXT
            )
        """)
        # Table for storing individual prompt-response items inside a session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()

def create_new_session(username: str, title: str = "New Chat"):
    session_id = f"sess_{int(datetime.now().timestamp())}"
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute(
            "INSERT INTO chat_sessions VALUES (?, ?, ?, ?)",
            (session_id, username, title, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
    return session_id

def get_user_sessions(username: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, session_title FROM chat_sessions WHERE username = ? ORDER BY created_at DESC", (username,))
        return cursor.fetchall()

def save_chat_message(session_id: str, role: str, content: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.cursor().execute(
            "INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()

def load_chat_messages(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
        return [{"role": r, "content": c} for r, c in cursor.fetchall()]
