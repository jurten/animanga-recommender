from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from pgvector import Vector

from . import db
from .config import get_settings


def embedding_text(item: dict[str, Any], traits: dict[str, Any]) -> str:
    payload = db.loads(item.get("payload_json"), {})
    alt_titles = payload.get("alternative_titles") or {}
    genres = [genre.get("name", "") for genre in payload.get("genres", [])]
    length_bucket = "short"
    units = item.get("num_units") or 0
    if units >= 50:
        length_bucket = "long"
    elif units > 13:
        length_bucket = "medium"
    fields = [
        item.get("title") or "",
        alt_titles.get("en") or "",
        alt_titles.get("ja") or "",
        " ".join(alt_titles.get("synonyms") or []),
        item.get("synopsis") or "",
        " ".join(genres),
        " ".join(payload.get("themes") or []),
        item.get("content_type") or "",
        item.get("media_type") or "",
        payload.get("source") or "",
        f"length:{length_bucket}",
        " ".join(traits.get("moods", [])),
        traits.get("comfort_level", ""),
        traits.get("mental_effort", ""),
        traits.get("emotional_load", ""),
        traits.get("depth", ""),
        traits.get("pacing", ""),
        traits.get("filler_risk", ""),
        traits.get("bingeability", ""),
        traits.get("cooldown_fit", ""),
        " ".join(traits.get("tags", [])),
    ]
    return "\n".join(part for part in fields if part)


def work_embedding_text(row: dict[str, Any]) -> str:
    traits = db.loads(row.get("traits_json"), {})
    titles = row.get("item_titles") or ""
    synopses = row.get("synopses") or ""
    roles = row.get("roles") or ""
    genres = row.get("genres") or ""
    return "\n".join(
        part
        for part in [
            row.get("work_title") or "",
            titles,
            synopses,
            genres,
            roles,
            " ".join(traits.get("moods", [])),
            traits.get("comfort_level", ""),
            traits.get("mental_effort", ""),
            traits.get("emotional_load", ""),
            traits.get("depth", ""),
            traits.get("pacing", ""),
            traits.get("filler_risk", ""),
            traits.get("bingeability", ""),
            traits.get("cooldown_fit", ""),
            " ".join(traits.get("tags", [])),
        ]
        if part
    )


def input_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_model():
    settings = get_settings()
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers to build local embeddings.") from exc
    return SentenceTransformer(settings.embedding_model, device=settings.embedding_device)


def _rows_needing_vectors(conn, limit: int | None = None) -> list[Any]:
    settings = get_settings()
    prompt_version = getattr(settings, "prompt_version", "traits-v1")
    sql = """
        SELECT
          i.id,
          i.content_type,
          i.title,
          i.synopsis,
          i.media_type,
          i.num_units,
          i.payload_json,
          t.traits_json
        FROM canonical_items i
        JOIN item_traits t ON t.canonical_item_id = i.id AND t.prompt_version = ?
        LEFT JOIN item_vectors v
          ON v.canonical_item_id = i.id AND v.model_name = ?
        WHERE v.canonical_item_id IS NULL OR v.input_hash != ?
        ORDER BY i.id
    """
    rows = conn.execute(sql, (prompt_version, settings.embedding_model, "__pending__")).fetchall()
    pending = []
    for row in rows:
        text = embedding_text(dict(row), db.loads(row["traits_json"], {}))
        existing = conn.execute(
            """
            SELECT input_hash FROM item_vectors
            WHERE canonical_item_id = ? AND model_name = ?
            """,
            (row["id"], settings.embedding_model),
        ).fetchone()
        if existing is None or existing["input_hash"] != input_hash(text):
            pending.append(row)
            if limit is not None and len(pending) >= limit:
                break
    return pending


def _work_rows_needing_vectors(conn, limit: int | None = None) -> list[Any]:
    settings = get_settings()
    if db.is_postgres_url():
        sql = """
            SELECT
              w.id AS work_id,
              w.title AS work_title,
              STRING_AGG(i.title, ' ') AS item_titles,
              STRING_AGG(COALESCE(i.synopsis, ''), ' ') AS synopses,
              STRING_AGG(l.role, ' ') AS roles,
              '{}' AS traits_json
            FROM canonical_works w
            JOIN item_work_links l ON l.canonical_work_id = w.id
            JOIN canonical_items i ON i.id = l.canonical_item_id
            GROUP BY w.id, w.title
            ORDER BY w.id
        """
    else:
        sql = """
            SELECT
              w.id AS work_id,
              w.title AS work_title,
              GROUP_CONCAT(i.title, ' ') AS item_titles,
              GROUP_CONCAT(COALESCE(i.synopsis, ''), ' ') AS synopses,
              GROUP_CONCAT(l.role, ' ') AS roles,
              '{}' AS traits_json
            FROM canonical_works w
            JOIN item_work_links l ON l.canonical_work_id = w.id
            JOIN canonical_items i ON i.id = l.canonical_item_id
            GROUP BY w.id, w.title
            ORDER BY w.id
        """
    rows = conn.execute(sql).fetchall()
    pending = []
    for row in rows:
        text = work_embedding_text(dict(row))
        existing = conn.execute(
            """
            SELECT input_hash FROM work_vectors
            WHERE canonical_work_id = ? AND model_name = ?
            """,
            (row["work_id"], settings.embedding_model),
        ).fetchone()
        if existing is None or existing["input_hash"] != input_hash(text):
            pending.append(row)
            if limit is not None and len(pending) >= limit:
                break
    return pending


def build_vectors(limit: int | None = None, level: str = "item", batch_size: int | None = None) -> int:
    settings = get_settings()
    effective_batch_size = batch_size or (
        settings.work_embedding_batch_size if level == "work" else settings.embedding_batch_size
    )
    effective_batch_size = max(1, effective_batch_size)
    count = 0
    with db.session() as conn:
        rows = _work_rows_needing_vectors(conn, limit) if level == "work" else _rows_needing_vectors(conn, limit)
        if not rows:
            return 0
        model = _load_model()
        for offset in range(0, len(rows), effective_batch_size):
            batch_rows = rows[offset : offset + effective_batch_size]
            if level == "work":
                batch_texts = [work_embedding_text(dict(row)) for row in batch_rows]
            else:
                batch_texts = [embedding_text(dict(row), db.loads(row["traits_json"], {})) for row in batch_rows]
            vectors = model.encode(batch_texts, normalize_embeddings=True)
            for row, text, vector in zip(batch_rows, batch_texts, vectors, strict=True):
                vector_list = [float(value) for value in vector]
                if level == "work" and db.is_postgres_url():
                    conn.execute(
                        """
                        INSERT INTO work_vectors (canonical_work_id, model_name, input_hash, embedding)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(canonical_work_id, model_name) DO UPDATE SET
                          input_hash = excluded.input_hash,
                          embedding = excluded.embedding,
                          created_at = now()
                        """,
                        (row["work_id"], settings.embedding_model, input_hash(text), Vector(vector_list)),
                    )
                elif level == "work":
                    conn.execute(
                        """
                        INSERT INTO work_vectors (canonical_work_id, model_name, input_hash, embedding_json)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(canonical_work_id, model_name) DO UPDATE SET
                          input_hash = excluded.input_hash,
                          embedding_json = excluded.embedding_json,
                          created_at = CURRENT_TIMESTAMP
                        """,
                        (row["work_id"], settings.embedding_model, input_hash(text), db.dumps(vector_list)),
                    )
                elif db.is_postgres_url():
                    conn.execute(
                        """
                        INSERT INTO item_vectors (canonical_item_id, model_name, input_hash, embedding)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(canonical_item_id, model_name) DO UPDATE SET
                          input_hash = excluded.input_hash,
                          embedding = excluded.embedding,
                          created_at = now()
                        """,
                        (row["id"], settings.embedding_model, input_hash(text), Vector(vector_list)),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO item_vectors (canonical_item_id, model_name, input_hash, embedding_json)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(canonical_item_id, model_name) DO UPDATE SET
                          input_hash = excluded.input_hash,
                          embedding_json = excluded.embedding_json,
                          created_at = CURRENT_TIMESTAMP
                        """,
                        (row["id"], settings.embedding_model, input_hash(text), db.dumps(vector_list)),
                    )
                count += 1
    return count


def query_vectors(query: str, limit: int = 20, level: str = "item") -> list[dict[str, Any]]:
    settings = get_settings()
    model = _load_model()
    vector = [float(value) for value in model.encode([query], normalize_embeddings=True)[0]]
    with db.session() as conn:
        if level == "work" and db.is_postgres_url():
            rows = conn.execute(
                """
                SELECT w.id, w.title, w.content_type, NULL AS source_item_id, 1 - (v.embedding <=> ?) AS score
                FROM work_vectors v
                JOIN canonical_works w ON w.id = v.canonical_work_id
                WHERE v.model_name = ?
                ORDER BY v.embedding <=> ?
                LIMIT ?
                """,
                (Vector(vector), settings.embedding_model, Vector(vector), limit),
            ).fetchall()
            return [dict(row) for row in rows]
        if db.is_postgres_url():
            rows = conn.execute(
                """
                SELECT i.id, i.title, i.content_type, mal.source_item_id, 1 - (v.embedding <=> ?) AS score
                FROM item_vectors v
                JOIN canonical_items i ON i.id = v.canonical_item_id
                LEFT JOIN item_source_links mal ON mal.canonical_item_id = i.id AND mal.source = 'mal'
                WHERE v.model_name = ?
                ORDER BY v.embedding <=> ?
                LIMIT ?
                """,
                (Vector(vector), settings.embedding_model, Vector(vector), limit),
            ).fetchall()
            return [dict(row) for row in rows]

        rows = conn.execute(
            """
            SELECT i.id, i.title, i.content_type, mal.source_item_id, v.embedding_json
            FROM item_vectors v
            JOIN canonical_items i ON i.id = v.canonical_item_id
            LEFT JOIN item_source_links mal ON mal.canonical_item_id = i.id AND mal.source = 'mal'
            WHERE v.model_name = ?
            """,
            (settings.embedding_model,),
        ).fetchall()
    query_vec = np.array(vector)
    scored = []
    for row in rows:
        item_vec = np.array(db.loads(row["embedding_json"], []))
        score = float(np.dot(query_vec, item_vec))
        result = dict(row)
        result.pop("embedding_json", None)
        result["score"] = score
        scored.append(result)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:limit]
