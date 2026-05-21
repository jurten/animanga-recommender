from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import get_settings


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  mal_user_id INTEGER,
  username TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mal_items (
  content_type TEXT NOT NULL,
  mal_id INTEGER NOT NULL,
  title TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  synopsis TEXT,
  media_type TEXT,
  status TEXT,
  mean REAL,
  rank INTEGER,
  popularity INTEGER,
  nsfw TEXT,
  num_units INTEGER,
  updated_at TEXT,
  PRIMARY KEY (content_type, mal_id)
);

CREATE TABLE IF NOT EXISTS user_list_entries (
  user_id INTEGER NOT NULL,
  content_type TEXT NOT NULL,
  mal_id INTEGER NOT NULL,
  status TEXT,
  score INTEGER,
  progress INTEGER,
  priority INTEGER,
  reconsume_count INTEGER,
  started_at TEXT,
  finished_at TEXT,
  updated_at TEXT,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (user_id, content_type, mal_id),
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (content_type, mal_id) REFERENCES mal_items(content_type, mal_id)
);

CREATE TABLE IF NOT EXISTS item_traits (
  content_type TEXT NOT NULL,
  mal_id INTEGER NOT NULL,
  prompt_version TEXT NOT NULL,
  model_name TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  traits_json TEXT NOT NULL,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (content_type, mal_id, prompt_version),
  FOREIGN KEY (content_type, mal_id) REFERENCES mal_items(content_type, mal_id)
);

CREATE TABLE IF NOT EXISTS recommendation_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  mode TEXT NOT NULL,
  mood TEXT,
  context_json TEXT NOT NULL,
  results_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS feedback_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER,
  user_id INTEGER NOT NULL,
  content_type TEXT NOT NULL,
  mal_id INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (run_id) REFERENCES recommendation_runs(id),
  FOREIGN KEY (user_id) REFERENCES users(id)
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or get_settings().sqlite_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: Path | None = None) -> None:
    with session(path) as conn:
        conn.executescript(SCHEMA)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)
