from __future__ import annotations

from fastapi import FastAPI, HTTPException

from . import db
from .models import FeedbackRequest, RecommendationRequest
from .recommender import recommend, record_feedback

app = FastAPI(title="MAL Recommender")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/items/{content_type}/{mal_id}")
def get_item(content_type: str, mal_id: int):
    with db.session() as conn:
        row = conn.execute(
            """
            SELECT i.*, mal.source_item_id AS mal_id, t.traits_json
            FROM item_source_links mal
            JOIN canonical_items i ON i.id = mal.canonical_item_id
            LEFT JOIN item_traits t ON t.canonical_item_id = i.id
            WHERE mal.source = 'mal' AND mal.content_type = ? AND mal.source_item_id = ?
            """,
            (content_type, str(mal_id)),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return {
        "content_type": row["content_type"],
        "mal_id": int(row["mal_id"]),
        "title": row["title"],
        "payload": db.loads(row["payload_json"], {}),
        "traits": db.loads(row["traits_json"], None),
    }


@app.post("/recommendations")
def create_recommendations(payload: RecommendationRequest):
    run_id, results = recommend(payload)
    return {"run_id": run_id, "results": [result.model_dump() for result in results]}


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    event_id = record_feedback(payload)
    return {"event_id": event_id}


@app.get("/runs/{run_id}")
def get_run(run_id: int):
    with db.session() as conn:
        row = conn.execute("SELECT * FROM recommendation_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "mode": row["mode"],
        "mood": row["mood"],
        "context": db.loads(row["context_json"], {}),
        "results": db.loads(row["results_json"], []),
        "created_at": row["created_at"],
    }
