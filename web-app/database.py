"""
SQLite 数据库 — 用户、知识库、会话记录
"""

import sqlite3
import hashlib
import secrets
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "app.db"


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """初始化表结构"""
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            kb_id INTEGER,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS agent_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            topic TEXT NOT NULL,
            log TEXT NOT NULL,
            report TEXT NOT NULL,
            created_at REAL NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    db.commit()
    db.close()


# ==================== 用户 ====================

def hash_password(password: str) -> str:
    salt = "ai-toolbox-salt"
    return hashlib.sha256((password + salt).encode()).hexdigest()


def create_user(username: str, password: str) -> dict:
    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, hash_password(password), time.time()),
        )
        db.commit()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return {"id": user["id"], "username": user["username"]}
    except sqlite3.IntegrityError:
        return {"error": "用户名已存在"}
    finally:
        db.close()


def login(username: str, password: str) -> dict:
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE username = ? AND password_hash = ?",
        (username, hash_password(password)),
    ).fetchone()

    if not user:
        db.close()
        return {"error": "用户名或密码错误"}

    token = secrets.token_hex(32)
    db.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user["id"], time.time()),
    )
    db.commit()
    db.close()
    return {"token": token, "user": {"id": user["id"], "username": user["username"]}}


def get_user_by_token(token: str) -> dict | None:
    db = get_db()
    row = db.execute(
        "SELECT u.* FROM users u JOIN sessions s ON u.id = s.user_id WHERE s.token = ?",
        (token,),
    ).fetchone()
    db.close()
    if row:
        return {"id": row["id"], "username": row["username"]}
    return None


# ==================== 知识库 ====================

def create_kb(user_id: int, name: str) -> dict:
    db = get_db()
    cur = db.execute(
        "INSERT INTO knowledge_bases (user_id, name, created_at) VALUES (?, ?, ?)",
        (user_id, name, time.time()),
    )
    db.commit()
    kb_id = cur.lastrowid
    db.close()
    return {"id": kb_id, "name": name}


def list_kbs(user_id: int) -> list:
    db = get_db()
    rows = db.execute(
        "SELECT id, name, created_at FROM knowledge_bases WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    db.close()
    return [{"id": r["id"], "name": r["name"], "created_at": r["created_at"]} for r in rows]


def delete_kb(user_id: int, kb_id: int) -> bool:
    db = get_db()
    db.execute("DELETE FROM knowledge_bases WHERE id = ? AND user_id = ?", (kb_id, user_id))
    db.commit()
    deleted = db.total_changes > 0
    db.close()
    return deleted


# ==================== 历史记录 ====================

def save_chat(user_id: int, role: str, content: str, kb_id: int = None):
    db = get_db()
    db.execute(
        "INSERT INTO chat_history (user_id, kb_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, kb_id, role, content, time.time()),
    )
    db.commit()
    db.close()


def get_chat_history(user_id: int, kb_id: int = None, limit: int = 50) -> list:
    db = get_db()
    if kb_id:
        rows = db.execute(
            "SELECT role, content, created_at FROM chat_history WHERE user_id = ? AND kb_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, kb_id, limit),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT role, content, created_at FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    db.close()
    return [{"role": r["role"], "content": r["content"], "time": r["created_at"]} for r in reversed(rows)]


def save_agent_log(user_id: int, topic: str, log: str, report: str) -> int:
    db = get_db()
    cur = db.execute(
        "INSERT INTO agent_logs (user_id, topic, log, report, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, topic, log, report, time.time()),
    )
    db.commit()
    log_id = cur.lastrowid
    db.close()
    return log_id


def get_agent_logs(user_id: int, limit: int = 20) -> list:
    db = get_db()
    rows = db.execute(
        "SELECT id, topic, report, created_at FROM agent_logs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    db.close()
    return [{"id": r["id"], "topic": r["topic"], "report": r["report"], "created_at": r["created_at"]} for r in rows]


def get_agent_log(user_id: int, log_id: int) -> dict | None:
    db = get_db()
    row = db.execute(
        "SELECT * FROM agent_logs WHERE id = ? AND user_id = ?",
        (log_id, user_id),
    ).fetchone()
    db.close()
    if row:
        return {"id": row["id"], "topic": row["topic"], "log": row["log"], "report": row["report"], "created_at": row["created_at"]}
    return None


# ==================== 初始化 ====================
init_db()
