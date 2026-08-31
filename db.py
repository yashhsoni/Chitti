import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")

SYSTEM_PROMPT = "You are Chitti the robo from enthiran, a helpful assistant. You can use available connectors/tools to answer user requests."


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                memory    TEXT NOT NULL UNIQUE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

def save_message(session_id: str, role: str, content: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )

def load_history(session_id: str) -> list:
    memories = load_memories()
    memory_block = ("\n\nThings the user has asked you to remember:\n" + "\n".join(f"- {m['memory']}" for m in memories)) if memories else ""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
    history = [{"role": "system", "content": SYSTEM_PROMPT + memory_block}]
    history += [{"role": row[0], "content": row[1]} for row in rows]
    return history

def save_memory(memory: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO memories (memory) VALUES (?)",
            (memory.strip(),)
        )

def load_memories() -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, memory FROM memories ORDER BY id ASC").fetchall()
    return [{"id": r[0], "memory": r[1]} for r in rows]


def delete_memory(memory_id: int):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))


def delete_session(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


def clear_history(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))


def get_all_sessions() -> list:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT m.session_id,
                   MIN(CASE WHEN m.role='user' THEN m.content END) as title,
                   MAX(m.timestamp) as last_active
            FROM messages m
            GROUP BY m.session_id
            ORDER BY last_active DESC
        """).fetchall()
    return [{"session_id": r[0], "title": (r[1] or "New Chat")[:60], "last_active": r[2]} for r in rows]
