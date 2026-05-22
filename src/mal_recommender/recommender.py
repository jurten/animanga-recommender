from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pgvector import Vector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import db
from .config import get_settings
from .models import RecommendationMode, RecommendationRequest, RecommendationResult
from .vectors import _load_model


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
    list_status: str | None
    list_intent: str | None
    seen_status: str | None
    user_score: int | None
    mean: float | None
    popularity: int | None
    num_units: int | None
    unlocked: bool = False
    selection_reason: str | None = None
    vector_score: float | None = None

    @property
    def mal_id(self) -> int:
        return int(self.source_item_id or self.canonical_item_id)


@dataclass
class LatestListEvent:
    status: str | None
    score: int | None


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


def _effective_source_item_id(row) -> str | None:
    if row["source_item_id"]:
        return str(row["source_item_id"])
    payload = db.loads(row["payload_json"], {})
    id_mal = payload.get("idMal")
    return str(id_mal) if id_mal else None


def _event_source_item_id(row) -> str | None:
    if row["source_item_id"]:
        return str(row["source_item_id"])
    payload = db.loads(row["payload_json"], {})
    node = payload.get("node") or {}
    node_id = node.get("id")
    return str(node_id) if node_id is not None else None


SUPPRESSED_RELEASE_ROLES = {"ova", "special", "recap", "spinoff", "alternative", "movie_extra"}
MAIN_RELEASE_ROLES = {None, "main", "sequel"}
SEEN_LIST_STATUSES = {"completed", "watching", "reading", "dropped"}
WISHLIST_LIST_STATUSES = {"plan_to_watch", "plan_to_read"}
ON_HOLD_LIST_STATUS = "on_hold"
WISHLIST_SCORE_BOOST = 0.35
ON_HOLD_SCORE_BOOST = 0.16
MOOD_ALIASES = {
    "psychological": {"psychological", "tense", "bleak", "thoughtful"},
    "cozy": {"cozy", "funny", "romantic"},
}


def _seen_status_from_list_status(status: str | None) -> str | None:
    return status if status in SEEN_LIST_STATUSES else None


def _list_intent_from_status(status: str | None) -> str | None:
    if status in WISHLIST_LIST_STATUSES:
        return status
    if status == ON_HOLD_LIST_STATUS:
        return status
    return None


def _latest_mal_list_events(conn, user_id: int) -> tuple[dict[int, LatestListEvent], dict[tuple[str, str], LatestListEvent]]:
    by_canonical_id: dict[int, LatestListEvent] = {}
    by_source_id: dict[tuple[str, str], LatestListEvent] = {}
    rows = conn.execute(
        """
        SELECT e.canonical_item_id, e.content_type, e.status, e.score, e.payload_json, mal.source_item_id
        FROM user_item_events e
        LEFT JOIN item_source_links mal
          ON mal.canonical_item_id = e.canonical_item_id AND mal.source = 'mal'
        WHERE e.user_id = ? AND e.source = 'mal' AND e.status IS NOT NULL
        ORDER BY e.id
        """,
        (user_id,),
    ).fetchall()
    for row in rows:
        event = LatestListEvent(status=row["status"], score=row["score"])
        by_canonical_id[int(row["canonical_item_id"])] = event
        source_item_id = _event_source_item_id(row)
        if source_item_id:
            by_source_id[(row["content_type"], source_item_id)] = event
    return by_canonical_id, by_source_id


def _load_all_candidates(conn, request: RecommendationRequest) -> list[Candidate]:
    settings = get_settings()
    prompt_version = getattr(settings, "prompt_version", "traits-v1")
    placeholders = ",".join("?" for _ in request.content_types)
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
          mal.source_item_id
        FROM canonical_items i
        LEFT JOIN item_traits t
          ON t.prompt_version = ?
          AND (
            t.canonical_item_id = i.id
            OR (
              t.content_type = i.content_type
              AND t.mal_id = CAST((
                SELECT source_item_id FROM item_source_links
                WHERE canonical_item_id = i.id AND source = 'mal'
                LIMIT 1
              ) AS INTEGER)
            )
          )
        LEFT JOIN item_source_links mal
          ON mal.canonical_item_id = i.id AND mal.source = 'mal'
        LEFT JOIN item_work_links l
          ON l.canonical_item_id = i.id
        LEFT JOIN canonical_works w
          ON w.id = l.canonical_work_id
        WHERE i.content_type IN ({placeholders})
        """,
        [prompt_version, *request.content_types],
    ).fetchall()
    events_by_canonical_id, events_by_source_id = _latest_mal_list_events(conn, request.user_id)
    candidates: list[Candidate] = []
    for row in rows:
        source_item_id = _effective_source_item_id(row)
        event = events_by_canonical_id.get(int(row["canonical_item_id"]))
        if event is None and source_item_id is not None:
            event = events_by_source_id.get((row["content_type"], source_item_id))
        list_status = event.status if event else None
        candidates.append(
            Candidate(
                canonical_item_id=row["canonical_item_id"],
                canonical_work_id=row["canonical_work_id"],
                work_title=row["work_title"],
                release_role=row["release_role"],
                sequence_index=row["sequence_index"],
                is_entrypoint=bool(row["is_entrypoint"]),
                content_type=row["content_type"],
                source_item_id=source_item_id,
                title=row["title"],
                text=_item_text(row),
                traits=db.loads(row["traits_json"], {}),
                list_status=list_status,
                list_intent=_list_intent_from_status(list_status),
                seen_status=_seen_status_from_list_status(list_status),
                user_score=event.score if event else None,
                mean=row["mean"],
                popularity=row["popularity"],
                num_units=row["num_units"],
            )
        )
    return candidates


def _is_completed_or_highly_rated(item: Candidate) -> bool:
    return item.seen_status == "completed" or (item.user_score is not None and item.user_score >= 8)


def _main_continuity(item: Candidate) -> bool:
    return item.release_role in MAIN_RELEASE_ROLES


def _select_release_for_work(work_candidates: list[Candidate], request: RecommendationRequest) -> Candidate | None:
    ordered = sorted(work_candidates, key=lambda item: (item.sequence_index, item.canonical_item_id))
    main_releases = [item for item in ordered if _main_continuity(item)]
    entrypoint = next((item for item in ordered if item.is_entrypoint), None)

    completed_main = [item for item in main_releases if _is_completed_or_highly_rated(item)]
    if completed_main:
        unlocked_after = max(item.sequence_index for item in completed_main)
        next_unseen = next(
            (item for item in main_releases if item.sequence_index > unlocked_after and not item.seen_status),
            None,
        )
        if next_unseen is not None:
            next_unseen.unlocked = True
            next_unseen.selection_reason = "unlocked next sequel from history"
            return next_unseen

    has_exposure = any(item.seen_status or item.user_score is not None for item in ordered)
    if not has_exposure and entrypoint is not None:
        if entrypoint.release_role == "sequel":
            return None
        entrypoint.selection_reason = "selected work entrypoint"
        return entrypoint

    if request.include_seen:
        fallback = entrypoint or (main_releases[0] if main_releases else ordered[0])
        fallback.selection_reason = "selected work entrypoint" if fallback.is_entrypoint else None
        return fallback

    return None


def _select_work_candidates(
    candidates: list[Candidate],
    request: RecommendationRequest,
    work_scores: dict[int, float] | None = None,
) -> list[Candidate]:
    if not _has_work_links_for_candidates(candidates):
        return [
            item
            for item in candidates
            if item.release_role not in SUPPRESSED_RELEASE_ROLES and (request.include_seen or not item.seen_status)
        ]

    by_work: dict[int, list[Candidate]] = {}
    unlinked: list[Candidate] = []
    for candidate in candidates:
        if candidate.canonical_work_id is None:
            unlinked.append(candidate)
        else:
            by_work.setdefault(candidate.canonical_work_id, []).append(candidate)

    selected = [
        item
        for item in unlinked
        if item.release_role not in SUPPRESSED_RELEASE_ROLES and (request.include_seen or not item.seen_status)
    ]
    work_ids = list(work_scores) if work_scores is not None else list(by_work)
    for work_id in work_ids:
        work_candidates = by_work.get(work_id)
        if not work_candidates:
            continue
        pick = _select_release_for_work(work_candidates, request)
        if pick is None:
            continue
        if pick.release_role in SUPPRESSED_RELEASE_ROLES:
            continue
        pick.vector_score = work_scores.get(work_id) if work_scores is not None else None
        selected.append(pick)
    return selected


def _has_work_links_for_candidates(candidates: list[Candidate]) -> bool:
    return any(candidate.canonical_work_id is not None for candidate in candidates)


def _load_candidates(conn, request: RecommendationRequest) -> list[Candidate]:
    return _select_work_candidates(_load_all_candidates(conn, request), request)


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
        if score >= 8 or row["event_type"] in {"completed", "liked"}:
            liked.append(candidate.text)
        if score and score <= 5 or row["event_type"] in {"dropped", "disliked", "skipped"}:
            disliked.append(candidate.text)
    return " ".join(liked), " ".join(disliked)


def _list_intent_bonus(candidate: Candidate) -> tuple[float, str | None]:
    if candidate.list_intent == "plan_to_watch":
        return WISHLIST_SCORE_BOOST, "on your MAL plan-to-watch list"
    if candidate.list_intent == "plan_to_read":
        return WISHLIST_SCORE_BOOST, "on your MAL plan-to-read list"
    if candidate.list_intent == ON_HOLD_LIST_STATUS:
        return ON_HOLD_SCORE_BOOST, "on your MAL on-hold list"
    return 0.0, None


def _query_context_text(conn, request: RecommendationRequest) -> str:
    settings = get_settings()
    prompt_version = getattr(settings, "prompt_version", "traits-v1")
    parts: list[str] = []
    if request.mood:
        parts.append(request.mood)
        if request.mood == "cozy":
            parts.append("comfort warm gentle slice of life")
        if request.mood == "psychological":
            parts.append("psychological thriller suspense tense mystery mind games dark introspective")
    mode_hints = {
        RecommendationMode.cooldown_after_heavy: "cooldown after heavy light comforting low emotional load",
        RecommendationMode.low_commitment: "low commitment short easy watch",
        RecommendationMode.challenge_me: "challenging deep complex ambitious",
        RecommendationMode.avoid_filler_risk: "avoid filler low friction main story",
        RecommendationMode.similar_to_recent: "similar to recent history",
        RecommendationMode.mood_match: "mood match",
    }
    parts.append(mode_hints.get(request.mode, request.mode.value))

    if request.recent_mal_ids:
        placeholders = ",".join("?" for _ in request.recent_mal_ids)
        rows = conn.execute(
            f"""
            SELECT i.title, i.synopsis, COALESCE(t.traits_json, '{{}}') AS traits_json
            FROM item_source_links mal
            JOIN canonical_items i ON i.id = mal.canonical_item_id
            LEFT JOIN item_traits t ON t.canonical_item_id = i.id AND t.prompt_version = ?
            WHERE mal.source = 'mal' AND CAST(mal.source_item_id AS INTEGER) IN ({placeholders})
            """,
            [prompt_version, *request.recent_mal_ids],
        ).fetchall()
        for row in rows:
            traits = db.loads(row["traits_json"], {})
            parts.extend(
                [
                    row["title"] or "",
                    row["synopsis"] or "",
                    " ".join(traits.get("moods", [])),
                    " ".join(traits.get("tags", [])),
                ]
            )
    return " ".join(part for part in parts if part)


def _encode_query_text(text: str) -> list[float] | None:
    if not text.strip():
        return None
    try:
        model = _load_model()
        return [float(value) for value in model.encode([text], normalize_embeddings=True)[0]]
    except Exception:
        return None


def _work_vector_scores(conn, request: RecommendationRequest) -> dict[int, float]:
    if not conn.execute("SELECT 1 FROM work_vectors LIMIT 1").fetchone():
        return {}
    query_vector = _encode_query_text(_query_context_text(conn, request))
    if query_vector is None:
        return {}

    candidate_limit = min(max(request.limit * 8, request.limit), 400)
    placeholders = ",".join("?" for _ in request.content_types)
    settings = get_settings()
    if db.is_postgres_url():
        rows = conn.execute(
            f"""
            SELECT w.id AS canonical_work_id, 1 - (v.embedding <=> ?) AS score
            FROM work_vectors v
            JOIN canonical_works w ON w.id = v.canonical_work_id
            WHERE v.model_name = ? AND w.content_type IN ({placeholders})
            ORDER BY v.embedding <=> ?
            LIMIT ?
            """,
            [Vector(query_vector), settings.embedding_model, *request.content_types, Vector(query_vector), candidate_limit],
        ).fetchall()
        return {int(row["canonical_work_id"]): float(row["score"]) for row in rows}

    rows = conn.execute(
        f"""
        SELECT w.id AS canonical_work_id, v.embedding_json
        FROM work_vectors v
        JOIN canonical_works w ON w.id = v.canonical_work_id
        WHERE v.model_name = ? AND w.content_type IN ({placeholders})
        """,
        [settings.embedding_model, *request.content_types],
    ).fetchall()
    query = np.array(query_vector)
    scored: list[tuple[int, float]] = []
    for row in rows:
        vector = np.array(db.loads(row["embedding_json"], []))
        if vector.size != query.size:
            continue
        scored.append((int(row["canonical_work_id"]), float(np.dot(query, vector))))
    scored.sort(key=lambda item: item[1], reverse=True)
    return dict(scored[:candidate_limit])


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
    desired_moods = MOOD_ALIASES.get(request.mood or "", {request.mood} if request.mood else set())
    if request.mood and moods & desired_moods:
        bonus += 0.3
        reasons.append(f"matches mood: {request.mood}")
    elif request.mood and moods and "balanced" not in moods:
        bonus -= 0.12
        reasons.append("penalized for mood mismatch")
    if request.mood == "psychological" and (
        traits.get("depth") == "heavy"
        or traits.get("mental_effort") == "high"
        or traits.get("emotional_load") == "high"
    ):
        bonus += 0.18
        reasons.append("psychological, more intense pick")
    if request.mood == "psychological" and (
        traits.get("comfort_level") == "high" or traits.get("cooldown_fit") == "high"
    ):
        bonus -= 0.18
        reasons.append("penalized because psychological mood avoids comfort picks")
    if request.mood == "cozy" and (
        traits.get("depth") == "heavy"
        or traits.get("mental_effort") == "high"
        or traits.get("emotional_load") == "high"
    ):
        bonus -= 0.16
        reasons.append("penalized because cozy mood avoids heavy picks")
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
        all_candidates = _load_all_candidates(conn, request)
        if not all_candidates:
            return 0, []
        work_scores = _work_vector_scores(conn, request)
        candidates = _select_work_candidates(all_candidates, request, work_scores or None)
        if not candidates:
            return 0, []

        liked_text, disliked_text = _taste_profile(conn, all_candidates, request.user_id)
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
            if candidate.vector_score is not None:
                score += candidate.vector_score
            trait_bonus, reasons = _trait_bonus(candidate, request, filler_endurance)
            intent_bonus, intent_reason = _list_intent_bonus(candidate)
            score += intent_bonus
            if intent_reason is not None:
                reasons.append(intent_reason)
            if candidate.vector_score is not None:
                reasons.insert(0, "matched semantic work vector")
            if candidate.selection_reason and candidate.selection_reason not in reasons:
                reasons.append(candidate.selection_reason)
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
