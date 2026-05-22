from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import db
from .models import RecommendationMode, RecommendationRequest, RecommendationResult


@dataclass
class Candidate:
    canonical_item_id: int
    canonical_work_id: int | None
    work_title: str | None
    release_role: str | None
    sequence_index: int
    is_entrypoint: bool
    content_type: str
    source_item_id: str | None
    title: str
    text: str
    traits: dict[str, Any]
    seen_status: str | None
    user_score: int | None
    mean: float | None
    popularity: int | None
    num_units: int | None
    unlocked: bool = False

    @property
    def mal_id(self) -> int:
        return int(self.source_item_id or self.canonical_item_id)


def _item_text(row) -> str:
    payload = db.loads(row["payload_json"], {})
    genres = " ".join(g.get("name", "") for g in payload.get("genres", []))
    alt_titles = payload.get("alternative_titles") or {}
    synonyms = " ".join(alt_titles.get("synonyms") or [])
    traits = db.loads(row["traits_json"], {})
    return " ".join(
        [
            row["title"] or "",
            alt_titles.get("en") or "",
            alt_titles.get("ja") or "",
            synonyms,
            row["synopsis"] or "",
            genres,
            row["media_type"] or "",
            payload.get("source") or "",
            " ".join(traits.get("moods", [])),
            traits.get("depth", ""),
            traits.get("pacing", ""),
            traits.get("emotional_load", ""),
            traits.get("mental_effort", ""),
            traits.get("comfort_level", ""),
            traits.get("commitment_cost", ""),
            traits.get("filler_risk", ""),
            traits.get("bingeability", ""),
            traits.get("cooldown_fit", ""),
            " ".join(traits.get("tags", [])),
        ]
    )


def _load_candidates(conn, request: RecommendationRequest) -> list[Candidate]:
    placeholders = ",".join("?" for _ in request.content_types)
    has_work_links = bool(conn.execute("SELECT 1 FROM item_work_links LIMIT 1").fetchone())
    rows = conn.execute(
        f"""
        SELECT
          i.id AS canonical_item_id,
          w.id AS canonical_work_id,
          w.title AS work_title,
          l.role AS release_role,
          COALESCE(l.sequence_index, 0) AS sequence_index,
          COALESCE(l.is_entrypoint, FALSE) AS is_entrypoint,
          i.content_type,
          i.title,
          i.synopsis,
          i.media_type,
          i.mean,
          i.popularity,
          i.num_units,
          i.payload_json,
          COALESCE(t.traits_json, '{{}}') AS traits_json,
          mal.source_item_id,
          seen.status AS seen_status,
          seen.score AS user_score
        FROM canonical_items i
        LEFT JOIN item_traits t
          ON t.canonical_item_id = i.id
          OR (
            t.content_type = i.content_type
            AND t.mal_id = CAST((
              SELECT source_item_id FROM item_source_links
              WHERE canonical_item_id = i.id AND source = 'mal'
              LIMIT 1
            ) AS INTEGER)
          )
        LEFT JOIN item_source_links mal
          ON mal.canonical_item_id = i.id AND mal.source = 'mal'
        LEFT JOIN item_work_links l
          ON l.canonical_item_id = i.id
        LEFT JOIN canonical_works w
          ON w.id = l.canonical_work_id
        LEFT JOIN (
          SELECT user_id, canonical_item_id, MAX(status) AS status, MAX(score) AS score, MAX(created_at) AS created_at
          FROM user_item_events
          WHERE user_id = ?
          GROUP BY user_id, canonical_item_id
        ) seen ON seen.canonical_item_id = i.id
        WHERE i.content_type IN ({placeholders})
        """,
        [request.user_id, *request.content_types],
    ).fetchall()
    candidates = [
        Candidate(
            canonical_item_id=row["canonical_item_id"],
            canonical_work_id=row["canonical_work_id"],
            work_title=row["work_title"],
            release_role=row["release_role"],
            sequence_index=row["sequence_index"],
            is_entrypoint=bool(row["is_entrypoint"]),
            content_type=row["content_type"],
            source_item_id=row["source_item_id"],
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
    if not has_work_links:
        return candidates

    by_work: dict[int, list[Candidate]] = {}
    unlinked: list[Candidate] = []
    for candidate in candidates:
        if candidate.canonical_work_id is None:
            unlinked.append(candidate)
        else:
            by_work.setdefault(candidate.canonical_work_id, []).append(candidate)

    selected = unlinked[:]
    for work_candidates in by_work.values():
        ordered = sorted(work_candidates, key=lambda item: (item.sequence_index, item.canonical_item_id))
        completed_seen = [
            item
            for item in ordered
            if item.seen_status == "completed" or (item.user_score is not None and item.user_score >= 8)
        ]
        if completed_seen:
            unseen_main = [
                item
                for item in ordered
                if not item.seen_status and item.release_role in {None, "main", "movie_extra", "sequel"}
            ]
            pick = unseen_main[0] if unseen_main else ordered[0]
            pick.unlocked = True
            selected.append(pick)
        else:
            entrypoint = next((item for item in ordered if item.is_entrypoint), ordered[0])
            selected.append(entrypoint)
    return [
        item
        for item in selected
        if request.include_seen or item.release_role not in {"ova", "special", "recap", "sequel", "spinoff"} or item.unlocked
    ]


def _taste_profile(conn, candidates: list[Candidate], user_id: int) -> tuple[str, str]:
    liked = []
    disliked = []
    by_id = {c.canonical_item_id: c for c in candidates}
    rows = conn.execute(
        """
        SELECT canonical_item_id, event_type, status, score
        FROM user_item_events
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchall()
    for row in rows:
        candidate = by_id.get(row["canonical_item_id"])
        if candidate is None:
            continue
        score = row["score"] or 0
        if score >= 8 or row["event_type"] in {"completed", "liked", "saved"}:
            liked.append(candidate.text)
        if score and score <= 5 or row["event_type"] in {"dropped", "disliked", "skipped"}:
            disliked.append(candidate.text)
    return " ".join(liked), " ".join(disliked)


def infer_filler_endurance(conn, user_id: int) -> float:
    rows = conn.execute(
        """
        SELECT e.status, e.progress, i.num_units
        FROM user_item_events e
        JOIN canonical_items i ON i.id = e.canonical_item_id
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
        conn.execute(
            """
            INSERT INTO users (id, username)
            VALUES (?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (request.user_id, f"user-{request.user_id}"),
        )
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
                    canonical_work_id=candidate.canonical_work_id,
                    work_title=candidate.work_title,
                    canonical_item_id=candidate.canonical_item_id,
                    release_role=candidate.release_role,
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
        canonical_item_id = None
        row = conn.execute(
            """
            SELECT canonical_item_id FROM item_source_links
            WHERE source = 'mal' AND content_type = ? AND source_item_id = ?
            """,
            (payload.content_type, str(payload.mal_id)),
        ).fetchone()
        if row:
            canonical_item_id = int(row["canonical_item_id"])
        cursor = conn.execute(
            """
            INSERT INTO feedback_events (
              run_id, user_id, canonical_item_id, content_type, source, source_item_id, event_type, payload_json
            )
            VALUES (?, ?, ?, ?, 'mal', ?, ?, ?)
            """,
            (
                payload.run_id,
                payload.user_id,
                canonical_item_id,
                payload.content_type,
                str(payload.mal_id),
                payload.event_type,
                db.dumps(payload.payload),
            ),
        )
        if canonical_item_id is not None:
            conn.execute(
                """
                INSERT INTO user_item_events (
                  user_id, canonical_item_id, content_type, event_type, source, source_event_id, payload_json
                )
                VALUES (?, ?, ?, ?, 'manual', ?, ?)
                """,
                (
                    payload.user_id,
                    canonical_item_id,
                    payload.content_type,
                    payload.event_type,
                    f"feedback:{cursor.lastrowid}",
                    db.dumps(payload.payload),
                ),
            )
        return int(cursor.lastrowid)
