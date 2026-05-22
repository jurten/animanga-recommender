from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import db


SAME_WORK_RELATIONS = {"prequel", "sequel", "parent", "side_story", "summary"}
EXTRA_RELATIONS = {"side_story", "summary"}
SUPPRESSED_ROLES = {"recap", "special", "ova"}


@dataclass
class WorkBuildResult:
    works: int
    links: int
    relations: int


def _norm_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"\bthe\s+movie\b", "", title)
    title = re.sub(r"\b(season|part)\s+\d+\b", "", title)
    title = re.sub(r"\b(ii|iii|iv|v|2nd|3rd|4th|second|third)\b", "", title)
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split()).strip()


def _is_extra_format(media_type: str | None) -> bool:
    return (media_type or "").lower() in {"ova", "special"}


def _role_for_item(item: dict[str, Any], relation_types: set[str]) -> str:
    media_type = (item.get("media_type") or "").lower()
    title = (item.get("title") or "").lower()
    if "summary" in relation_types or "recap" in title:
        return "recap"
    if "missing pieces" in title:
        return "special"
    if media_type == "ova":
        return "ova"
    if media_type == "special":
        return "special"
    if re.search(r"\b(cm|pv|trailer|special|ova)\b", title):
        return "special"
    if media_type == "movie" and "the movie" in title:
        return "movie_extra"
    if re.search(r"\b(season\s*[2-9]|[2-9](nd|rd|th)\s+season|ii|iii|iv|v|too!?|[2-9])\b$", title):
        return "sequel"
    if media_type == "movie" and relation_types & EXTRA_RELATIONS:
        return "movie_extra"
    if media_type in {"tv", "tv_short"}:
        return "main"
    if relation_types & {"alternative"}:
        return "alternative"
    if relation_types & {"spin_off", "character"}:
        return "spinoff"
    return "main"


def _sort_key(item: dict[str, Any]) -> tuple[str, int, int]:
    payload = db.loads(item.get("payload_json"), {})
    start_date = payload.get("start_date") or "9999-99-99"
    popularity = item.get("popularity") or 999999999
    return start_date, popularity, item["id"]


class UnionFind:
    def __init__(self, ids: list[int]):
        self.parent = {item_id: item_id for item_id in ids}

    def find(self, item_id: int) -> int:
        parent = self.parent[item_id]
        if parent != item_id:
            self.parent[item_id] = self.find(parent)
        return self.parent[item_id]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def build_works(content_type: str = "anime", path: Path | None = None) -> WorkBuildResult:
    with db.session(path) as conn:
        items = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, content_type, title, media_type, popularity, payload_json
                FROM canonical_items
                WHERE content_type = ?
                """,
                (content_type,),
            ).fetchall()
        ]
        if not items:
            return WorkBuildResult(works=0, links=0, relations=0)

        item_by_id = {item["id"]: item for item in items}
        relation_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT from_item_id, to_item_id, relation_type
                FROM item_relations
                WHERE from_item_id IN (SELECT id FROM canonical_items WHERE content_type = ?)
                  AND to_item_id IN (SELECT id FROM canonical_items WHERE content_type = ?)
                """,
                (content_type, content_type),
            ).fetchall()
        ]

        uf = UnionFind(list(item_by_id))
        relation_types_by_item: dict[int, set[str]] = defaultdict(set)
        for relation in relation_rows:
            relation_type = relation["relation_type"]
            relation_types_by_item[relation["from_item_id"]].add(relation_type)
            relation_types_by_item[relation["to_item_id"]].add(relation_type)
            if relation_type in SAME_WORK_RELATIONS:
                uf.union(relation["from_item_id"], relation["to_item_id"])

        # Fallback title grouping catches common sequels already in the DB before
        # richer relation ingestion has been run.
        by_title: dict[str, list[int]] = defaultdict(list)
        for item in items:
            key = _norm_title(item["title"])
            if key:
                by_title[key].append(item["id"])
        for ids in by_title.values():
            if len(ids) > 1:
                first = ids[0]
                for item_id in ids[1:]:
                    uf.union(first, item_id)

        components: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            components[uf.find(item["id"])].append(item)

        conn.execute("DELETE FROM work_vectors WHERE canonical_work_id IN (SELECT id FROM canonical_works WHERE content_type = ?)", (content_type,))
        conn.execute("DELETE FROM item_work_links WHERE canonical_work_id IN (SELECT id FROM canonical_works WHERE content_type = ?)", (content_type,))
        conn.execute("DELETE FROM canonical_works WHERE content_type = ?", (content_type,))

        works = 0
        links = 0
        for component in components.values():
            ordered = sorted(component, key=_sort_key)
            entry_candidates = [
                item
                for item in ordered
                if not _is_extra_format(item.get("media_type"))
                and _role_for_item(item, relation_types_by_item[item["id"]]) not in SUPPRESSED_ROLES
            ]
            entrypoint = entry_candidates[0] if entry_candidates else ordered[0]
            title = entrypoint["title"]
            cursor = conn.execute(
                """
                INSERT INTO canonical_works (content_type, title, payload_json)
                VALUES (?, ?, ?)
                """,
                (
                    content_type,
                    title,
                    db.dumps(
                        {
                            "entrypoint_item_id": entrypoint["id"],
                            "normalized_title": _norm_title(title),
                            "item_count": len(ordered),
                        }
                    ),
                ),
            )
            work_id = int(cursor.lastrowid)
            works += 1
            for idx, item in enumerate(ordered):
                relation_types = sorted(relation_types_by_item[item["id"]])
                role = _role_for_item(item, set(relation_types))
                conn.execute(
                    """
                    INSERT INTO item_work_links (
                      canonical_item_id, canonical_work_id, role, sequence_index,
                      is_entrypoint, confidence, source, evidence_json
                    )
                    VALUES (?, ?, ?, ?, ?, 0.85, 'derived', ?)
                    ON CONFLICT(canonical_item_id) DO UPDATE SET
                      canonical_work_id = excluded.canonical_work_id,
                      role = excluded.role,
                      sequence_index = excluded.sequence_index,
                      is_entrypoint = excluded.is_entrypoint,
                      confidence = excluded.confidence,
                      source = excluded.source,
                      evidence_json = excluded.evidence_json,
                      updated_at = CURRENT_TIMESTAMP
                    """,
                    (
                        item["id"],
                        work_id,
                        role,
                        idx,
                        item["id"] == entrypoint["id"],
                        db.dumps({"relation_types": relation_types, "normalized_title": _norm_title(item["title"])}),
                    ),
                )
                links += 1

        return WorkBuildResult(works=works, links=links, relations=len(relation_rows))
