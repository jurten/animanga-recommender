from __future__ import annotations

import hashlib
from typing import Any

from . import db
from .anilist import AniListClient, normalize_anilist_media, normalize_anilist_relation_type
from .mal import MALClient


HISTORY_STATUS_EVENTS = {
    "completed": "completed",
    "watching": "watched",
    "reading": "read",
    "plan_to_watch": "saved",
    "plan_to_read": "saved",
    "dropped": "dropped",
    "on_hold": "saved",
}


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(db.dumps(payload).encode()).hexdigest()


def _unit_count(content_type: str, node: dict[str, Any]) -> int | None:
    if content_type == "anime":
        return node.get("num_episodes")
    return node.get("num_chapters") or node.get("num_volumes")


def _source_item_id(source_item_id: int | str) -> str:
    return str(source_item_id)


def _mirror_legacy_tables(conn) -> bool:
    return not isinstance(conn, db.PostgresConnection)


def upsert_user(conn, user_payload: dict[str, Any]) -> int:
    username = user_payload.get("name") or "me"
    mal_user_id = user_payload.get("id")
    conn.execute(
        """
        INSERT INTO users (id, mal_user_id, username)
        VALUES (?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET mal_user_id = excluded.mal_user_id, username = excluded.username
        """,
        (1, mal_user_id, username),
    )
    return 1


def insert_source_record(
    conn,
    source: str,
    content_type: str,
    source_item_id: int | str,
    payload: dict[str, Any],
    source_updated_at: str | None = None,
) -> int:
    payload_hash = _payload_hash(payload)
    conn.execute(
        """
        INSERT INTO source_records (
          source, content_type, source_item_id, payload_hash, payload_json, source_updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, content_type, source_item_id, payload_hash) DO NOTHING
        """,
        (source, content_type, _source_item_id(source_item_id), payload_hash, db.dumps(payload), source_updated_at),
    )
    row = conn.execute(
        """
        SELECT id FROM source_records
        WHERE source = ? AND content_type = ? AND source_item_id = ? AND payload_hash = ?
        """,
        (source, content_type, _source_item_id(source_item_id), payload_hash),
    ).fetchone()
    return int(row["id"])


def upsert_canonical_item(
    conn,
    source: str,
    content_type: str,
    source_item_id: int | str,
    node: dict[str, Any],
    source_record_id: int | None = None,
) -> int:
    existing = conn.execute(
        """
        SELECT canonical_item_id FROM item_source_links
        WHERE source = ? AND content_type = ? AND source_item_id = ?
        """,
        (source, content_type, _source_item_id(source_item_id)),
    ).fetchone()
    values = (
        content_type,
        node["title"],
        node.get("synopsis"),
        node.get("media_type"),
        node.get("status"),
        node.get("mean"),
        node.get("rank"),
        node.get("popularity"),
        node.get("nsfw"),
        _unit_count(content_type, node),
        db.dumps(node),
        node.get("updated_at"),
    )
    if existing:
        canonical_item_id = int(existing["canonical_item_id"])
        conn.execute(
            """
            UPDATE canonical_items
            SET content_type = ?, title = ?, synopsis = ?, media_type = ?, status = ?,
                mean = ?, rank = ?, popularity = ?, nsfw = ?, num_units = ?,
                payload_json = ?, updated_at = COALESCE(?, CURRENT_TIMESTAMP)
            WHERE id = ?
            """,
            (*values, canonical_item_id),
        )
    else:
        cursor = conn.execute(
            """
            INSERT INTO canonical_items (
              content_type, title, synopsis, media_type, status, mean, rank,
              popularity, nsfw, num_units, payload_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
            """,
            values,
        )
        canonical_item_id = int(cursor.lastrowid)

    conn.execute(
        """
        INSERT INTO item_source_links (
          canonical_item_id, source, content_type, source_item_id, source_record_id
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source, content_type, source_item_id) DO UPDATE SET
          canonical_item_id = excluded.canonical_item_id,
          source_record_id = excluded.source_record_id,
          updated_at = CURRENT_TIMESTAMP
        """,
        (canonical_item_id, source, content_type, _source_item_id(source_item_id), source_record_id),
    )
    for field_name in ["title", "synopsis", "media_type", "status", "mean", "rank", "popularity", "num_units"]:
        if node.get(field_name) is not None or field_name == "num_units":
            conn.execute(
                """
                INSERT INTO item_field_sources (
                  canonical_item_id, field_name, source, source_record_id, confidence
                )
                VALUES (?, ?, ?, ?, 1.0)
                ON CONFLICT(canonical_item_id, field_name) DO UPDATE SET
                  source = excluded.source,
                  source_record_id = excluded.source_record_id,
                  confidence = excluded.confidence,
                  updated_at = CURRENT_TIMESTAMP
                """,
                (canonical_item_id, field_name, source, source_record_id),
            )
    return canonical_item_id


def upsert_item(conn, content_type: str, node: dict[str, Any]) -> int:
    source_record_id = insert_source_record(conn, "mal", content_type, node["id"], node, node.get("updated_at"))
    canonical_item_id = upsert_canonical_item(conn, "mal", content_type, node["id"], node, source_record_id)
    if _mirror_legacy_tables(conn):
        conn.execute(
            """
            INSERT INTO mal_items (
              content_type, mal_id, title, payload_json, synopsis, media_type, status,
              mean, rank, popularity, nsfw, num_units, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(content_type, mal_id) DO UPDATE SET
              title = excluded.title,
              payload_json = excluded.payload_json,
              synopsis = excluded.synopsis,
              media_type = excluded.media_type,
              status = excluded.status,
              mean = excluded.mean,
              rank = excluded.rank,
              popularity = excluded.popularity,
              nsfw = excluded.nsfw,
              num_units = excluded.num_units,
              updated_at = excluded.updated_at
            """,
            (
                content_type,
                node["id"],
                node["title"],
                db.dumps(node),
                node.get("synopsis"),
                node.get("media_type"),
                node.get("status"),
                node.get("mean"),
                node.get("rank"),
                node.get("popularity"),
                node.get("nsfw"),
                _unit_count(content_type, node),
                node.get("updated_at"),
            ),
        )
    return canonical_item_id


def canonical_id_for_source(conn, source: str, content_type: str, source_item_id: int | str) -> int | None:
    row = conn.execute(
        """
        SELECT canonical_item_id FROM item_source_links
        WHERE source = ? AND content_type = ? AND source_item_id = ?
        """,
        (source, content_type, _source_item_id(source_item_id)),
    ).fetchone()
    return int(row["canonical_item_id"]) if row else None


def upsert_item_relation(
    conn,
    from_item_id: int,
    to_item_id: int,
    relation_type: str,
    source: str,
    evidence: dict[str, Any] | None = None,
    confidence: float = 1.0,
) -> None:
    if from_item_id == to_item_id:
        return
    conn.execute(
        """
        INSERT INTO item_relations (
          from_item_id, to_item_id, relation_type, source, confidence, evidence_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(from_item_id, to_item_id, relation_type, source) DO UPDATE SET
          confidence = excluded.confidence,
          evidence_json = excluded.evidence_json
        """,
        (from_item_id, to_item_id, relation_type, source, confidence, db.dumps(evidence or {})),
    )


def add_history_event(
    conn,
    user_id: int,
    canonical_item_id: int,
    content_type: str,
    event_type: str,
    source: str,
    payload: dict[str, Any] | None = None,
    *,
    source_event_id: str | None = None,
    status: str | None = None,
    score: int | None = None,
    progress: int | None = None,
    occurred_at: str | None = None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO user_item_events (
          user_id, canonical_item_id, content_type, event_type, source, source_event_id,
          status, score, progress, payload_json, occurred_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
        """,
        (
            user_id,
            canonical_item_id,
            content_type,
            event_type,
            source,
            source_event_id,
            status,
            score,
            progress,
            db.dumps(payload or {}),
            occurred_at,
        ),
    )
    return int(cursor.lastrowid)


def upsert_entry(conn, user_id: int, content_type: str, edge: dict[str, Any]) -> None:
    node = edge["node"]
    canonical_item_id = canonical_id_for_source(conn, "mal", content_type, node["id"]) or upsert_item(conn, content_type, node)
    status = edge.get("list_status") or node.get("my_list_status") or {}
    progress = status.get("num_episodes_watched") if content_type == "anime" else status.get("num_chapters_read")
    event_type = HISTORY_STATUS_EVENTS.get(status.get("status"), "rated" if status.get("score") else "saved")
    source_event_id = f"mal:{user_id}:{content_type}:{node['id']}:{status.get('updated_at') or status.get('status') or 'list'}"
    add_history_event(
        conn,
        user_id,
        canonical_item_id,
        content_type,
        event_type,
        "mal",
        edge,
        source_event_id=source_event_id,
        status=status.get("status"),
        score=status.get("score"),
        progress=progress,
        occurred_at=status.get("updated_at"),
    )
    if _mirror_legacy_tables(conn):
        conn.execute(
            """
            INSERT INTO user_list_entries (
              user_id, content_type, mal_id, status, score, progress, priority,
              reconsume_count, started_at, finished_at, updated_at, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, content_type, mal_id) DO UPDATE SET
              status = excluded.status,
              score = excluded.score,
              progress = excluded.progress,
              priority = excluded.priority,
              reconsume_count = excluded.reconsume_count,
              started_at = excluded.started_at,
              finished_at = excluded.finished_at,
              updated_at = excluded.updated_at,
              payload_json = excluded.payload_json
            """,
            (
                user_id,
                content_type,
                node["id"],
                status.get("status"),
                status.get("score"),
                progress,
                status.get("priority"),
                status.get("num_times_rewatched") or status.get("num_times_reread"),
                status.get("start_date"),
                status.get("finish_date"),
                status.get("updated_at"),
                db.dumps(edge),
            ),
        )


async def ingest_user_list(content_type: str, client: MALClient | None = None) -> int:
    client = client or MALClient()
    user_payload = await client.get_user()
    count = 0
    with db.session() as conn:
        run = conn.execute(
            """
            INSERT INTO ingestion_runs (source, job_type, status)
            VALUES ('mal', ?, 'running')
            """,
            (f"{content_type}-list",),
        )
        run_id = int(run.lastrowid)
        user_id = upsert_user(conn, user_payload)
        try:
            async for edge in client.iter_user_list(content_type):
                upsert_item(conn, content_type, edge["node"])
                upsert_entry(conn, user_id, content_type, edge)
                count += 1
            conn.execute(
                """
                UPDATE ingestion_runs
                SET status = 'succeeded', items_seen = ?, items_written = ?, finished_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (count, count, run_id),
            )
        except Exception as exc:
            conn.execute(
                """
                UPDATE ingestion_runs
                SET status = 'failed', error = ?, items_seen = ?, items_written = ?, finished_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (str(exc), count, count, run_id),
            )
            raise
    return count


async def ingest_anilist(content_type: str, limit: int = 100, client: AniListClient | None = None) -> int:
    client = client or AniListClient()
    count = 0
    page = 1
    with db.session() as conn:
        run = conn.execute(
            """
            INSERT INTO ingestion_runs (source, job_type, status)
            VALUES ('anilist', ?, 'running')
            """,
            (f"{content_type}-metadata",),
        )
        run_id = int(run.lastrowid)
        try:
            while count < limit:
                payload = await client.media_page(content_type, page=page, per_page=min(50, limit - count))
                for media in payload.get("media", []):
                    node = normalize_anilist_media(media)
                    source_record_id = insert_source_record(
                        conn, "anilist", content_type, media["id"], media, node.get("updated_at")
                    )
                    canonical_item_id = None
                    if node.get("idMal"):
                        canonical_item_id = canonical_id_for_source(conn, "mal", content_type, node["idMal"])
                    if canonical_item_id is None:
                        canonical_item_id = upsert_canonical_item(
                            conn, "anilist", content_type, media["id"], node, source_record_id
                        )
                    if node.get("idMal"):
                        conn.execute(
                            """
                            INSERT INTO item_source_links (
                              canonical_item_id, source, content_type, source_item_id, source_record_id
                            )
                            VALUES (?, 'mal', ?, ?, NULL)
                            ON CONFLICT(source, content_type, source_item_id) DO UPDATE SET
                              canonical_item_id = excluded.canonical_item_id,
                              updated_at = CURRENT_TIMESTAMP
                            """,
                            (canonical_item_id, content_type, str(node["idMal"])),
                        )
                    conn.execute(
                        """
                        INSERT INTO item_source_links (
                          canonical_item_id, source, content_type, source_item_id, source_record_id
                        )
                        VALUES (?, 'anilist', ?, ?, ?)
                        ON CONFLICT(source, content_type, source_item_id) DO UPDATE SET
                          canonical_item_id = excluded.canonical_item_id,
                          source_record_id = excluded.source_record_id,
                          updated_at = CURRENT_TIMESTAMP
                        """,
                        (canonical_item_id, content_type, str(media["id"]), source_record_id),
                    )
                    for edge in (media.get("relations") or {}).get("edges") or []:
                        related_media = edge.get("node") or {}
                        if related_media.get("type") not in {"ANIME", "MANGA"}:
                            continue
                        related_content_type = "anime" if related_media.get("type") == "ANIME" else "manga"
                        if related_content_type != content_type:
                            continue
                        related_node = normalize_anilist_media(related_media)
                        related_record_id = insert_source_record(
                            conn,
                            "anilist",
                            related_content_type,
                            related_media["id"],
                            related_media,
                            related_node.get("updated_at"),
                        )
                        related_item_id = None
                        if related_node.get("idMal"):
                            related_item_id = canonical_id_for_source(
                                conn, "mal", related_content_type, related_node["idMal"]
                            )
                        if related_item_id is None:
                            related_item_id = upsert_canonical_item(
                                conn,
                                "anilist",
                                related_content_type,
                                related_media["id"],
                                related_node,
                                related_record_id,
                            )
                        if related_node.get("idMal"):
                            conn.execute(
                                """
                                INSERT INTO item_source_links (
                                  canonical_item_id, source, content_type, source_item_id, source_record_id
                                )
                                VALUES (?, 'mal', ?, ?, NULL)
                                ON CONFLICT(source, content_type, source_item_id) DO UPDATE SET
                                  canonical_item_id = excluded.canonical_item_id,
                                  updated_at = CURRENT_TIMESTAMP
                                """,
                                (related_item_id, related_content_type, str(related_node["idMal"])),
                            )
                        conn.execute(
                            """
                            INSERT INTO item_source_links (
                              canonical_item_id, source, content_type, source_item_id, source_record_id
                            )
                            VALUES (?, 'anilist', ?, ?, ?)
                            ON CONFLICT(source, content_type, source_item_id) DO UPDATE SET
                              canonical_item_id = excluded.canonical_item_id,
                              source_record_id = excluded.source_record_id,
                              updated_at = CURRENT_TIMESTAMP
                            """,
                            (related_item_id, related_content_type, str(related_media["id"]), related_record_id),
                        )
                        upsert_item_relation(
                            conn,
                            canonical_item_id,
                            related_item_id,
                            normalize_anilist_relation_type(edge.get("relationType")),
                            "anilist",
                            {"relationType": edge.get("relationType"), "anilist_id": related_media["id"]},
                        )
                    count += 1
                    if count >= limit:
                        break
                if not payload.get("pageInfo", {}).get("hasNextPage"):
                    break
                page += 1
            conn.execute(
                """
                UPDATE ingestion_runs
                SET status = 'succeeded', cursor = ?, items_seen = ?, items_written = ?, finished_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (str(page), count, count, run_id),
            )
        except Exception as exc:
            conn.execute(
                """
                UPDATE ingestion_runs
                SET status = 'failed', cursor = ?, items_seen = ?, items_written = ?, error = ?, finished_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (str(page), count, count, str(exc), run_id),
            )
            raise
    return count
