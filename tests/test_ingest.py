from mal_recommender import db
from mal_recommender.ingest import upsert_entry, upsert_item, upsert_user


def test_upsert_anime_entry(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    edge = {
        "node": {
            "id": 1,
            "title": "Test Anime",
            "synopsis": "A bright comedy.",
            "genres": [{"name": "Comedy"}],
            "media_type": "tv",
            "num_episodes": 12,
        },
        "list_status": {
            "status": "completed",
            "score": 9,
            "num_episodes_watched": 12,
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    }
    with db.session(path) as conn:
        user_id = upsert_user(conn, {"id": 10, "name": "tester"})
        upsert_item(conn, "anime", edge["node"])
        upsert_entry(conn, user_id, "anime", edge)
        row = conn.execute("SELECT * FROM user_list_entries").fetchone()

    assert row["status"] == "completed"
    assert row["score"] == 9
    assert row["progress"] == 12
