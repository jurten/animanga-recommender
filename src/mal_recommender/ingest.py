from __future__ import annotations

from typing import Any

from . import db
from .mal import MALClient


def _unit_count(content_type: str, node: dict[str, Any]) -> int | None:
    if content_type == "anime":
        return node.get("num_episodes")
    return node.get("num_chapters") or node.get("num_volumes")


def upsert_user(conn, user_payload: dict[str, Any]) -> int:
    username = user_payload.get("name") or "me"
    mal_user_id = user_payload.get("id")
    conn.execute(
        """
        INSERT INTO users (id, mal_user_id, username)
        VALUES (1, ?, ?)
        ON CONFLICT(id) DO UPDATE SET mal_user_id = excluded.mal_user_id, username = excluded.username
        """,
        (mal_user_id, username),
    )
    return 1


def upsert_item(conn, content_type: str, node: dict[str, Any]) -> None:
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


def upsert_entry(conn, user_id: int, content_type: str, edge: dict[str, Any]) -> None:
    node = edge["node"]
    status = edge.get("list_status") or {}
    progress = status.get("num_episodes_watched") if content_type == "anime" else status.get("num_chapters_read")
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
        user_id = upsert_user(conn, user_payload)
        async for edge in client.iter_user_list(content_type):
            upsert_item(conn, content_type, edge["node"])
            upsert_entry(conn, user_id, content_type, edge)
            count += 1
    return count
