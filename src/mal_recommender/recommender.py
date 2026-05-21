from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import db
from .models import RecommendationMode, RecommendationRequest, RecommendationResult


@dataclass
class Candidate:
    content_type: str
    mal_id: int
    title: str
    text: str
    traits: dict[str, Any]
    seen_status: str | None
    user_score: int | None
    mean: float | None
    popularity: int | None
    num_units: int | None


def _item_text(row) -> str:
    payload = db.loads(row["payload_json"], {})
    genres = " ".join(g.get("name", "") for g in payload.get("genres", []))
    traits = db.loads(row["traits_json"], {})
    return " ".join(
        [
            row["title"] or "",
            row["synopsis"] or "",
            genres,
            row["media_type"] or "",
            " ".join(traits.get("moods", [])),
            traits.get("depth", ""),
            traits.get("pacing", ""),
            traits.get("emotional_load", ""),
            traits.get("commitment_cost", ""),
            traits.get("filler_risk", ""),
            traits.get("bingeability", ""),
            traits.get("cooldown_fit", ""),
            " ".join(traits.get("tags", [])),
        ]
    )


def _load_candidates(conn, request: RecommendationRequest) -> list[Candidate]:
    placeholders = ",".join("?" for _ in request.content_types)
    rows = conn.execute(
        f"""
        SELECT i.*, t.traits_json, e.status AS seen_status, e.score AS user_score
        FROM mal_items i
        JOIN item_traits t ON t.content_type = i.content_type AND t.mal_id = i.mal_id
        LEFT JOIN user_list_entries e
          ON e.content_type = i.content_type
         AND e.mal_id = i.mal_id
         AND e.user_id = ?
        WHERE i.content_type IN ({placeholders})
        """,
        [request.user_id, *request.content_types],
    ).fetchall()
    return [
        Candidate(
            content_type=row["content_type"],
            mal_id=row["mal_id"],
            title=row["title"],
            text=_item_text(row),
            traits=db.loads(row["traits_json"], {}),
            seen_status=row["seen_status"],
            user_score=row["user_score"],
            mean=row["mean"],
            popularity=row["popularity"],
            num_units=row["num_units"],
        )
        for row in rows
    ]


def _taste_profile(conn, candidates: list[Candidate], user_id: int) -> tuple[str, str]:
    liked = []
    disliked = []
    by_key = {(c.content_type, c.mal_id): c for c in candidates}
    rows = conn.execute(
        """
        SELECT content_type, mal_id, status, score
        FROM user_list_entries
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchall()
    for row in rows:
        candidate = by_key.get((row["content_type"], row["mal_id"]))
        if candidate is None:
            continue
        score = row["score"] or 0
        if score >= 8 or row["status"] == "completed":
            liked.append(candidate.text)
        if score and score <= 5 or row["status"] == "dropped":
            disliked.append(candidate.text)
    return " ".join(liked), " ".join(disliked)


def infer_filler_endurance(conn, user_id: int) -> float:
    rows = conn.execute(
        """
        SELECT e.status, e.progress, i.num_units
        FROM user_list_entries e
        JOIN mal_items i ON i.content_type = e.content_type AND i.mal_id = e.mal_id
        WHERE e.user_id = ? AND i.num_units IS NOT NULL AND i.num_units > 0
        """,
        (user_id,),
    ).fetchall()
    if not rows:
        return 0.5
    long_completed = 0
    long_started = 0
    early_drops = 0
    for row in rows:
        units = row["num_units"] or 0
        progress = row["progress"] or 0
        if units >= 50 and progress > 0:
            long_started += 1
        if units >= 50 and row["status"] == "completed":
            long_completed += 1
        if row["status"] == "dropped" and progress / units < 0.25:
            early_drops += 1
    endurance = 0.5
    endurance += min(0.35, long_completed * 0.08)
    endurance -= min(0.35, early_drops * 0.08)
    if long_started:
        endurance += min(0.15, (long_completed / long_started) * 0.15)
    return float(max(0.0, min(1.0, endurance)))


def _trait_bonus(candidate: Candidate, request: RecommendationRequest, filler_endurance: float) -> tuple[float, list[str]]:
    traits = candidate.traits
    reasons: list[str] = []
    bonus = 0.0
    moods = set(traits.get("moods", []))
    if request.mood and request.mood in moods:
        bonus += 0.18
        reasons.append(f"matches mood: {request.mood}")
    if request.mode == RecommendationMode.cooldown_after_heavy and traits.get("cooldown_fit") == "high":
        bonus += 0.22
        reasons.append("good cooldown after heavier material")
    if request.mode == RecommendationMode.cooldown_after_heavy and (
        traits.get("depth") == "heavy" or traits.get("emotional_load") == "high"
    ):
        bonus -= 0.18
        reasons.append("penalized because cooldown mode avoids heavy picks")
    if request.mode == RecommendationMode.low_commitment and traits.get("commitment_cost") == "low":
        bonus += 0.18
        reasons.append("low commitment")
    if request.mode == RecommendationMode.challenge_me and traits.get("depth") == "heavy":
        bonus += 0.15
        reasons.append("heavier, more challenging pick")
    if request.mode == RecommendationMode.avoid_filler_risk and traits.get("filler_risk") == "low":
        bonus += 0.18
        reasons.append("low filler/friction risk")
    if traits.get("filler_risk") == "high" and filler_endurance < 0.45:
        bonus -= 0.2
        reasons.append("penalized for low filler endurance")
    if traits.get("filler_risk") == "high" and filler_endurance > 0.7:
        bonus += 0.08
        reasons.append("fits high filler endurance")
    return bonus, reasons


def recommend(request: RecommendationRequest) -> tuple[int, list[RecommendationResult]]:
    with db.session() as conn:
        candidates = _load_candidates(conn, request)
        if not candidates:
            return 0, []

        liked_text, disliked_text = _taste_profile(conn, candidates, request.user_id)
        corpus = [c.text for c in candidates] + [liked_text or "liked anime manga", disliked_text or "dropped disliked"]
        vectorizer = TfidfVectorizer(max_features=6000, stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
        candidate_matrix = matrix[: len(candidates)]
        liked_vec = matrix[len(candidates)]
        disliked_vec = matrix[len(candidates) + 1]
        liked_scores = cosine_similarity(candidate_matrix, liked_vec).ravel()
        disliked_scores = cosine_similarity(candidate_matrix, disliked_vec).ravel()
        filler_endurance = infer_filler_endurance(conn, request.user_id)

        scored: list[RecommendationResult] = []
        for idx, candidate in enumerate(candidates):
            if candidate.seen_status and not request.include_seen:
                continue
            score = float(liked_scores[idx] - (0.45 * disliked_scores[idx]))
            trait_bonus, reasons = _trait_bonus(candidate, request, filler_endurance)
            score += trait_bonus
            if candidate.mean:
                score += min(candidate.mean / 100.0, 0.1)
            if candidate.popularity:
                score += min(1 / candidate.popularity, 0.05)
            if not reasons:
                reasons.append("close to established taste profile")
            scored.append(
                RecommendationResult(
                    content_type=candidate.content_type,
                    mal_id=candidate.mal_id,
                    title=candidate.title,
                    score=round(score, 4),
                    reasons=reasons,
                    traits=candidate.traits,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        results = scored[: request.limit]
        cursor = conn.execute(
            """
            INSERT INTO recommendation_runs (user_id, mode, mood, context_json, results_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                request.user_id,
                request.mode.value,
                request.mood,
                db.dumps({"filler_endurance": filler_endurance, "content_types": request.content_types}),
                db.dumps([result.model_dump() for result in results]),
            ),
        )
        return int(cursor.lastrowid), results


def record_feedback(payload) -> int:
    with db.session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO feedback_events (run_id, user_id, content_type, mal_id, event_type, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                payload.run_id,
                payload.user_id,
                payload.content_type,
                payload.mal_id,
                payload.event_type,
                db.dumps(payload.payload),
            ),
        )
        return int(cursor.lastrowid)
