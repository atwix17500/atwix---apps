import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "crop_app.db"
USERS_JSON = Path(__file__).parent.parent / "data" / "users.json"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    nitrogen REAL NOT NULL,
    phosphorus REAL NOT NULL,
    potassium REAL NOT NULL,
    temperature REAL NOT NULL,
    humidity REAL NOT NULL,
    ph REAL NOT NULL,
    rainfall REAL NOT NULL,
    recommended_crop TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
"""


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate_schema(conn)
    _migrate_from_json(conn)


def _migrate_schema(conn: sqlite3.Connection):
    columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "country" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN country TEXT NOT NULL DEFAULT ''")
    if "district" not in columns:
        conn.execute("ALTER TABLE users ADD COLUMN district TEXT NOT NULL DEFAULT ''")
    conn.commit()


def _migrate_from_json(conn: sqlite3.Connection):
    if not USERS_JSON.exists():
        return

    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count > 0:
        return

    with open(USERS_JSON, "r", encoding="utf-8") as f:
        users = json.load(f)

    for username, data in users.items():
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash, role, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                username.lower(),
                data.get("email", "").lower(),
                data["password_hash"],
                data.get("role", "user"),
                data.get("created_at", ""),
            ),
        )

    conn.commit()
