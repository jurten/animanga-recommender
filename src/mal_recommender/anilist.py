from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx


API_URL = "https://graphql.anilist.co"


MEDIA_QUERY = """
query ($page: Int!, $perPage: Int!, $type: MediaType) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    media(type: $type, sort: POPULARITY_DESC) {
      id
      idMal
      type
      format
      status
      title { romaji english native }
      synonyms
      description(asHtml: false)
      genres
      tags { name rank isMediaSpoiler }
      episodes
      chapters
      volumes
      averageScore
      popularity
      source
      updatedAt
      relations {
        edges {
          relationType
          node {
            id
            idMal
            type
            format
            status
            title { romaji english native }
            synonyms
            description(asHtml: false)
            genres
            tags { name rank isMediaSpoiler }
            episodes
            chapters
            volumes
            averageScore
            popularity
            source
            updatedAt
          }
        }
      }
    }
  }
}
"""


class AniListClient:
    async def media_page(self, content_type: str, page: int = 1, per_page: int = 50) -> dict[str, Any]:
        media_type = "ANIME" if content_type == "anime" else "MANGA"
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                API_URL,
                json={"query": MEDIA_QUERY, "variables": {"page": page, "perPage": per_page, "type": media_type}},
            )
            response.raise_for_status()
            return response.json()["data"]["Page"]


def normalize_anilist_media(media: dict[str, Any]) -> dict[str, Any]:
    title = media.get("title") or {}
    content_type = "anime" if media.get("type") == "ANIME" else "manga"
    tags = [
        tag["name"]
        for tag in media.get("tags") or []
        if tag.get("rank", 0) >= 50 and not tag.get("isMediaSpoiler")
    ]
    updated_at = None
    if media.get("updatedAt"):
        updated_at = datetime.fromtimestamp(media["updatedAt"], tz=UTC).isoformat()
    return {
        "id": media["id"],
        "idMal": media.get("idMal"),
        "title": title.get("english") or title.get("romaji") or title.get("native") or str(media["id"]),
        "alternative_titles": {
            "en": title.get("english"),
            "ja": title.get("native"),
            "synonyms": [item for item in media.get("synonyms") or [] if item],
        },
        "synopsis": media.get("description"),
        "genres": [{"name": genre} for genre in media.get("genres") or []],
        "themes": tags,
        "media_type": (media.get("format") or "").lower(),
        "status": (media.get("status") or "").lower(),
        "mean": (media.get("averageScore") / 10) if media.get("averageScore") else None,
        "popularity": media.get("popularity"),
        "num_episodes": media.get("episodes") if content_type == "anime" else None,
        "num_chapters": media.get("chapters") if content_type == "manga" else None,
        "num_volumes": media.get("volumes") if content_type == "manga" else None,
        "source": (media.get("source") or "").lower(),
        "updated_at": updated_at,
    }


ANILIST_RELATION_TYPES = {
    "PREQUEL": "prequel",
    "SEQUEL": "sequel",
    "PARENT": "parent",
    "SIDE_STORY": "side_story",
    "SUMMARY": "summary",
    "ALTERNATIVE": "alternative",
    "SPIN_OFF": "spin_off",
    "CHARACTER": "character",
    "OTHER": "other",
}


def normalize_anilist_relation_type(value: str | None) -> str:
    return ANILIST_RELATION_TYPES.get(value or "", (value or "other").lower())
