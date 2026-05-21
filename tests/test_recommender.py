from mal_recommender import db
from mal_recommender.ingest import upsert_entry, upsert_item, upsert_user
from mal_recommender.models import RecommendationMode, RecommendationRequest
from mal_recommender.recommender import infer_filler_endurance, recommend
from mal_recommender.traits import heuristic_traits, source_hash


def _seed_item(conn, content_type, mal_id, title, genres, synopsis, episodes=12, status=None, score=None):
    node = {
        "id": mal_id,
        "title": title,
        "synopsis": synopsis,
        "genres": [{"name": genre} for genre in genres],
        "media_type": "tv",
        "num_episodes": episodes if content_type == "anime" else None,
        "num_chapters": episodes if content_type == "manga" else None,
    }
    upsert_item(conn, content_type, node)
    label = heuristic_traits(node)
    conn.execute(
        """
        INSERT INTO item_traits (
          content_type, mal_id, prompt_version, model_name, source_hash, traits_json, confidence
        )
        VALUES (?, ?, 'traits-v1', 'heuristic-v1', ?, ?, ?)
        """,
        (content_type, mal_id, source_hash(node), db.dumps(label.model_dump()), label.confidence),
    )
    if status:
        edge = {
            "node": node,
            "list_status": {
                "status": status,
                "score": score,
                "num_episodes_watched": episodes if content_type == "anime" else None,
                "num_chapters_read": episodes if content_type == "manga" else None,
            },
        }
        upsert_entry(conn, 1, content_type, edge)


def test_recommend_prefers_cooldown_item(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: type("S", (), {"sqlite_path": path})())
    with db.session(path) as conn:
        upsert_user(conn, {"id": 1, "name": "tester"})
        _seed_item(conn, "anime", 1, "Heavy Seen", ["Drama"], "A story about grief and war.", 12, "completed", 9)
        _seed_item(conn, "anime", 2, "Cozy Pick", ["Comedy", "Slice of Life"], "Friends relax and tell jokes.", 12)
        _seed_item(conn, "anime", 3, "Bleak Pick", ["Drama"], "War and death shape everyone.", 12)

    request = RecommendationRequest(
        user_id=1,
        mode=RecommendationMode.cooldown_after_heavy,
        mood="cozy",
        content_types=["anime"],
        limit=2,
    )
    run_id, results = recommend(request)

    assert run_id == 1
    assert results[0].title == "Cozy Pick"


def test_filler_endurance_penalizes_early_drops(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    with db.session(path) as conn:
        upsert_user(conn, {"id": 1, "name": "tester"})
        node = {
            "id": 10,
            "title": "Long Show",
            "synopsis": "",
            "genres": [],
            "media_type": "tv",
            "num_episodes": 120,
        }
        upsert_item(conn, "anime", node)
        upsert_entry(
            conn,
            1,
            "anime",
            {"node": node, "list_status": {"status": "dropped", "score": 4, "num_episodes_watched": 5}},
        )
        endurance = infer_filler_endurance(conn, 1)

    assert endurance < 0.5
