from __future__ import annotations

import hashlib
import json
from typing import Any

from . import db
from .config import get_settings
from .models import TraitLabel


MOOD_BY_GENRE = {
    "Comedy": "funny",
    "Slice of Life": "cozy",
    "Iyashikei": "cozy",
    "Action": "hype",
    "Adventure": "adventurous",
    "Drama": "thoughtful",
    "Psychological": "tense",
    "Suspense": "tense",
    "Horror": "bleak",
    "Romance": "romantic",
    "Sports": "hype",
}


def source_hash(item: dict[str, Any]) -> str:
    payload = json.dumps(item, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def heuristic_traits(item: dict[str, Any]) -> TraitLabel:
    genres = [genre.get("name", "") for genre in item.get("genres", [])]
    synopsis = (item.get("synopsis") or "").lower()
    moods = sorted({MOOD_BY_GENRE[g] for g in genres if g in MOOD_BY_GENRE})
    if not moods:
        moods = ["thoughtful"] if "life" in synopsis or "mystery" in synopsis else ["balanced"]

    heavy_terms = ["death", "trauma", "war", "grief", "despair", "abuse", "murder", "suicide"]
    depth = "heavy" if any(term in synopsis for term in heavy_terms) else "moderate"
    if any(g in genres for g in ["Comedy", "Slice of Life", "Iyashikei"]):
        depth = "shallow" if depth != "heavy" else "moderate"

    units = item.get("num_episodes") or item.get("num_chapters") or item.get("num_volumes") or 0
    commitment = "low" if units and units <= 13 else "medium"
    if units and units >= 50:
        commitment = "high"

    filler_risk = "high" if units and units >= 100 else "medium"
    if item.get("media_type") in {"movie", "one_shot"} or commitment == "low":
        filler_risk = "low"

    pacing = "fast" if any(g in genres for g in ["Action", "Sports"]) else "balanced"
    if commitment == "high" and "Slice of Life" in genres:
        pacing = "slow"

    emotional = "high" if depth == "heavy" else "low" if depth == "shallow" else "moderate"
    cooldown = "high" if depth == "shallow" and emotional != "high" else "low" if depth == "heavy" else "medium"
    bingeability = "high" if commitment in {"low", "medium"} and filler_risk != "high" else "medium"

    return TraitLabel(
        moods=moods,
        depth=depth,
        pacing=pacing,
        emotional_load=emotional,
        commitment_cost=commitment,
        filler_risk=filler_risk,
        bingeability=bingeability,
        cooldown_fit=cooldown,
        tags=genres,
        confidence=0.55,
        rationale="Heuristic label from MAL genres, synopsis terms, media type, and length.",
    )


async def llm_traits(item: dict[str, Any]) -> TraitLabel | None:
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return None

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    prompt = {
        "title": item.get("title"),
        "synopsis": item.get("synopsis"),
        "genres": [g.get("name") for g in item.get("genres", [])],
        "media_type": item.get("media_type"),
        "status": item.get("status"),
        "num_episodes": item.get("num_episodes"),
        "num_chapters": item.get("num_chapters"),
        "num_volumes": item.get("num_volumes"),
    }
    response = await client.responses.create(
        model=settings.openai_model,
        input=[
            {
                "role": "system",
                "content": "Return compact JSON matching this schema: moods array, depth, pacing, emotional_load, commitment_cost, filler_risk, bingeability, cooldown_fit, tags array, confidence number, rationale string.",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )
    text = response.output_text
    return TraitLabel.model_validate_json(text)


async def label_missing(limit: int = 200, use_llm: bool = True) -> int:
    settings = get_settings()
    count = 0
    with db.session() as conn:
        rows = conn.execute(
            """
            SELECT content_type, mal_id, payload_json
            FROM mal_items
            WHERE NOT EXISTS (
              SELECT 1 FROM item_traits
              WHERE item_traits.content_type = mal_items.content_type
                AND item_traits.mal_id = mal_items.mal_id
                AND item_traits.prompt_version = ?
            )
            LIMIT ?
            """,
            (settings.prompt_version, limit),
        ).fetchall()
        for row in rows:
            item = db.loads(row["payload_json"], {})
            label = await llm_traits(item) if use_llm else None
            if label is None:
                label = heuristic_traits(item)
                model_name = "heuristic-v1"
            else:
                model_name = settings.openai_model
            conn.execute(
                """
                INSERT OR REPLACE INTO item_traits (
                  content_type, mal_id, prompt_version, model_name, source_hash,
                  traits_json, confidence
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["content_type"],
                    row["mal_id"],
                    settings.prompt_version,
                    model_name,
                    source_hash(item),
                    db.dumps(label.model_dump()),
                    label.confidence,
                ),
            )
            count += 1
    return count
