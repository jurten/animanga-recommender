from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


ContentType = Literal["anime", "manga"]


class RecommendationMode(StrEnum):
    similar_to_recent = "similar_to_recent"
    mood_match = "mood_match"
    cooldown_after_heavy = "cooldown_after_heavy"
    low_commitment = "low_commitment"
    challenge_me = "challenge_me"
    avoid_filler_risk = "avoid_filler_risk"


class TraitLabel(BaseModel):
    moods: list[str] = Field(default_factory=list)
    depth: str = "moderate"
    pacing: str = "balanced"
    emotional_load: str = "moderate"
    commitment_cost: str = "medium"
    filler_risk: str = "medium"
    bingeability: str = "medium"
    cooldown_fit: str = "medium"
    tags: list[str] = Field(default_factory=list)
    confidence: float = 0.6
    rationale: str = ""


class RecommendationRequest(BaseModel):
    user_id: int = 1
    mood: str | None = None
    mode: RecommendationMode = RecommendationMode.mood_match
    limit: int = Field(default=20, ge=1, le=100)
    include_seen: bool = False
    content_types: list[ContentType] = Field(default_factory=lambda: ["anime", "manga"])
    recent_mal_ids: list[int] = Field(default_factory=list)


class RecommendationResult(BaseModel):
    content_type: ContentType
    mal_id: int
    title: str
    score: float
    reasons: list[str]
    traits: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    run_id: int | None = None
    user_id: int = 1
    content_type: ContentType
    mal_id: int
    event_type: Literal["shown", "opened", "dismissed", "saved", "marked_not_now", "accepted"]
    payload: dict[str, Any] = Field(default_factory=dict)
