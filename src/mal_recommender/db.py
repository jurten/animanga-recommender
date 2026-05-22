from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .config import get_settings


SQLITE_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  mal_user_id INTEGER,
  username TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL,
  cursor TEXT,
  items_seen INTEGER NOT NULL DEFAULT 0,
  items_written INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS source_records (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source TEXT NOT NULL,
  content_type TEXT NOT NULL,
  source_item_id TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  source_updated_at TEXT,
  UNIQUE (source, content_type, source_item_id, payload_hash)
);

CREATE TABLE IF NOT EXISTS canonical_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content_type TEXT NOT NULL,
  title TEXT NOT NULL,
  synopsis TEXT,
  media_type TEXT,
  status TEXT,
  mean REAL,
  rank INTEGER,
  popularity INTEGER,
  nsfw TEXT,
  num_units INTEGER,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS item_source_links (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  canonical_item_id INTEGER NOT NULL,
  source TEXT NOT NULL,
  content_type TEXT NOT NULL,
  source_item_id TEXT NOT NULL,
  source_record_id INTEGER,
  url TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (source, content_type, source_item_id),
  FOREIGN KEY (canonical_item_id) REFERENCES canonical_items(id),
  FOREIGN KEY (source_record_id) REFERENCES source_records(id)
);

CREATE TABLE IF NOT EXISTS item_field_sources (
  canonical_item_id INTEGER NOT NULL,
  field_name TEXT NOT NULL,
  source TEXT NOT NULL,
  source_record_id INTEGER,
  confidence REAL NOT NULL DEFAULT 1.0,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (canonical_item_id, field_name),
  FOREIGN KEY (canonical_item_id) REFERENCES canonical_items(id),
  FOREIGN KEY (source_record_id) REFERENCES source_records(id)
);

CREATE TABLE IF NOT EXISTS user_item_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  canonical_item_id INTEGER NOT NULL,
  content_type TEXT NOT NULL,
  event_type TEXT NOT NULL,
  source TEXT NOT NULL,
  source_event_id TEXT,
  status TEXT,
  score INTEGER,
  progress INTEGER,
  payload_json TEXT NOT NULL DEFAULT '{}',
  occurred_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (canonical_item_id) REFERENCES canonical_items(id)
);

CREATE INDEX IF NOT EXISTS idx_user_item_events_user ON user_item_events (user_id, canonical_item_id);
CREATE INDEX IF NOT EXISTS idx_user_item_events_type ON user_item_events (event_type, source);

CREATE TABLE IF NOT EXISTS item_traits (
  canonical_item_id INTEGER,
  content_type TEXT,
  mal_id INTEGER,
  prompt_version TEXT NOT NULL,
  model_name TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  traits_json TEXT NOT NULL,
  confidence REAL NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (canonical_item_id) REFERENCES canonical_items(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_item_traits_canonical_unique
  ON item_traits (canonical_item_id, prompt_version)
;

CREATE UNIQUE INDEX IF NOT EXISTS idx_item_traits_mal
  ON item_traits (content_type, mal_id, prompt_version)
  WHERE content_type IS NOT NULL AND mal_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS item_vectors (
  canonical_item_id INTEGER NOT NULL,
  model_name TEXT NOT NULL,
  input_hash TEXT NOT NULL,
  embedding_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (canonical_item_id, model_name),
  FOREIGN KEY (canonical_item_id) REFERENCES canonical_items(id)
);

CREATE TABLE IF NOT EXISTS canonical_works (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  content_type TEXT NOT NULL,
  title TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS item_relations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  from_item_id INTEGER NOT NULL,
  to_item_id INTEGER NOT NULL,
  relation_type TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 1.0,
  evidence_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (from_item_id, to_item_id, relation_type, source),
  FOREIGN KEY (from_item_id) REFERENCES canonical_items(id),
  FOREIGN KEY (to_item_id) REFERENCES canonical_items(id)
);

CREATE TABLE IF NOT EXISTS item_work_links (
  canonical_item_id INTEGER NOT NULL,
  canonical_work_id INTEGER NOT NULL,
  role TEXT NOT NULL,
  sequence_index INTEGER NOT NULL DEFAULT 0,
  is_entrypoint INTEGER NOT NULL DEFAULT 0,
  confidence REAL NOT NULL DEFAULT 1.0,
  source TEXT NOT NULL,
  evidence_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (canonical_item_id),
  FOREIGN KEY (canonical_item_id) REFERENCES canonical_items(id),
  FOREIGN KEY (canonical_work_id) REFERENCES canonical_works(id)
);

CREATE INDEX IF NOT EXISTS idx_item_work_links_work ON item_work_links (canonical_work_id, sequence_index);

CREATE TABLE IF NOT EXISTS work_vectors (
  canonical_work_id INTEGER NOT NULL,
  model_name TEXT NOT NULL,
  input_hash TEXT NOT NULL,
  embedding_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (canonical_work_id, model_name),
  FOREIGN KEY (canonical_work_id) REFERENCES canonical_works(id)
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
  canonical_item_id INTEGER,
  content_type TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'manual',
  source_item_id TEXT,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (run_id) REFERENCES recommendation_runs(id),
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (canonical_item_id) REFERENCES canonical_items(id)
);

-- Compatibility tables for older tests and one-off SQLite databases. Runtime code
-- writes canonical tables first and mirrors MAL imports here during the transition.
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
  FOREIGN KEY (user_id) REFERENCES users(id)
);
"""


POSTGRES_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  mal_user_id BIGINT,
  username TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  source TEXT NOT NULL,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL,
  cursor TEXT,
  items_seen INTEGER NOT NULL DEFAULT 0,
  items_written INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS source_records (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  source TEXT NOT NULL,
  content_type TEXT NOT NULL,
  source_item_id TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json JSONB NOT NULL,
  fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  source_updated_at TIMESTAMPTZ,
  UNIQUE (source, content_type, source_item_id, payload_hash)
);

CREATE TABLE IF NOT EXISTS canonical_items (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  content_type TEXT NOT NULL,
  title TEXT NOT NULL,
  synopsis TEXT,
  media_type TEXT,
  status TEXT,
  mean DOUBLE PRECISION,
  rank INTEGER,
  popularity INTEGER,
  nsfw TEXT,
  num_units INTEGER,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS item_source_links (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  canonical_item_id BIGINT NOT NULL REFERENCES canonical_items(id),
  source TEXT NOT NULL,
  content_type TEXT NOT NULL,
  source_item_id TEXT NOT NULL,
  source_record_id BIGINT REFERENCES source_records(id),
  url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (source, content_type, source_item_id)
);

CREATE TABLE IF NOT EXISTS item_field_sources (
  canonical_item_id BIGINT NOT NULL REFERENCES canonical_items(id),
  field_name TEXT NOT NULL,
  source TEXT NOT NULL,
  source_record_id BIGINT REFERENCES source_records(id),
  confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (canonical_item_id, field_name)
);

CREATE TABLE IF NOT EXISTS user_item_events (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(id),
  canonical_item_id BIGINT NOT NULL REFERENCES canonical_items(id),
  content_type TEXT NOT NULL,
  event_type TEXT NOT NULL,
  source TEXT NOT NULL,
  source_event_id TEXT,
  status TEXT,
  score INTEGER,
  progress INTEGER,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_item_events_user ON user_item_events (user_id, canonical_item_id);
CREATE INDEX IF NOT EXISTS idx_user_item_events_type ON user_item_events (event_type, source);

CREATE TABLE IF NOT EXISTS item_traits (
  canonical_item_id BIGINT REFERENCES canonical_items(id),
  content_type TEXT,
  mal_id BIGINT,
  prompt_version TEXT NOT NULL,
  model_name TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  traits_json JSONB NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_item_traits_canonical_unique
  ON item_traits (canonical_item_id, prompt_version)
;

CREATE UNIQUE INDEX IF NOT EXISTS idx_item_traits_mal
  ON item_traits (content_type, mal_id, prompt_version)
  WHERE content_type IS NOT NULL AND mal_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS item_vectors (
  canonical_item_id BIGINT NOT NULL REFERENCES canonical_items(id),
  model_name TEXT NOT NULL,
  input_hash TEXT NOT NULL,
  embedding vector(1024) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (canonical_item_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_item_vectors_embedding
  ON item_vectors USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS canonical_works (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  content_type TEXT NOT NULL,
  title TEXT NOT NULL,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS item_relations (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  from_item_id BIGINT NOT NULL REFERENCES canonical_items(id),
  to_item_id BIGINT NOT NULL REFERENCES canonical_items(id),
  relation_type TEXT NOT NULL,
  source TEXT NOT NULL,
  confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (from_item_id, to_item_id, relation_type, source)
);

CREATE TABLE IF NOT EXISTS item_work_links (
  canonical_item_id BIGINT PRIMARY KEY REFERENCES canonical_items(id),
  canonical_work_id BIGINT NOT NULL REFERENCES canonical_works(id),
  role TEXT NOT NULL,
  sequence_index INTEGER NOT NULL DEFAULT 0,
  is_entrypoint BOOLEAN NOT NULL DEFAULT false,
  confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
  source TEXT NOT NULL,
  evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_item_work_links_work ON item_work_links (canonical_work_id, sequence_index);

CREATE TABLE IF NOT EXISTS work_vectors (
  canonical_work_id BIGINT NOT NULL REFERENCES canonical_works(id),
  model_name TEXT NOT NULL,
  input_hash TEXT NOT NULL,
  embedding vector(1024) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (canonical_work_id, model_name)
);

CREATE INDEX IF NOT EXISTS idx_work_vectors_embedding
  ON work_vectors USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS recommendation_runs (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(id),
  mode TEXT NOT NULL,
  mood TEXT,
  context_json JSONB NOT NULL,
  results_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feedback_events (
  id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  run_id BIGINT REFERENCES recommendation_runs(id),
  user_id BIGINT NOT NULL REFERENCES users(id),
  canonical_item_id BIGINT REFERENCES canonical_items(id),
  content_type TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'manual',
  source_item_id TEXT,
  event_type TEXT NOT NULL,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class PostgresConnection:
    def __init__(self, conn: Any):
        self._conn = conn

    def execute(self, sql: str, params: Any = None):
        translated = _postgres_sql(sql)
        wants_lastrowid = _postgres_insert_needs_returning(translated)
        if wants_lastrowid:
            translated = f"{translated.rstrip()} RETURNING id"
        cursor = self._conn.execute(translated, params)
        if wants_lastrowid:
            row = cursor.fetchone()
            return PostgresCursor(cursor, int(row["id"]))
        return cursor

    def executescript(self, script: str) -> None:
        for statement in _split_sql(script):
            self.execute(statement)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


def _split_sql(script: str) -> list[str]:
    return [statement.strip() for statement in script.split(";") if statement.strip()]


def _postgres_sql(sql: str) -> str:
    return sql.replace("?", "%s")


def _postgres_insert_needs_returning(sql: str) -> bool:
    normalized = " ".join(sql.lower().split())
    if not normalized.startswith("insert into") or " on conflict" in normalized or " returning " in normalized:
        return False
    identity_tables = {
        "users",
        "ingestion_runs",
        "source_records",
        "canonical_items",
        "item_source_links",
        "user_item_events",
        "recommendation_runs",
        "feedback_events",
        "canonical_works",
        "item_relations",
    }
    return any(normalized.startswith(f"insert into {table}") for table in identity_tables)


class PostgresCursor:
    def __init__(self, cursor: Any, lastrowid: int | None = None):
        self._cursor = cursor
        self.lastrowid = lastrowid

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


def is_postgres_url(database_url: str | None = None) -> bool:
    url = database_url or getattr(get_settings(), "database_url", "")
    return url.startswith(("postgresql://", "postgres://"))


def connect(path: Path | None = None):
    settings = get_settings()
    database_url = getattr(settings, "database_url", "")
    if path is None and is_postgres_url(database_url):
        try:
            import psycopg
            from psycopg.rows import dict_row
            from pgvector.psycopg import register_vector
        except ImportError as exc:
            raise RuntimeError("Install psycopg and pgvector to use a Postgres DATABASE_URL.") from exc
        pg_conn = psycopg.connect(database_url, row_factory=dict_row)
        try:
            has_vector_extension = bool(
                pg_conn.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector' LIMIT 1"
                ).fetchone()
            )
        except Exception:
            has_vector_extension = False
        if has_vector_extension:
            register_vector(pg_conn)
        return PostgresConnection(pg_conn)

    db_path = path or settings.sqlite_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session(path: Path | None = None) -> Iterator[Any]:
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
        conn.executescript(POSTGRES_SCHEMA if path is None and is_postgres_url() else SQLITE_SCHEMA)


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def loads(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)
