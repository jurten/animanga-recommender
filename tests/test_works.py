from mal_recommender import db
from mal_recommender.ingest import upsert_item, upsert_item_relation
from mal_recommender.models import RecommendationRequest
from mal_recommender.recommender import recommend
from mal_recommender.traits import heuristic_traits, source_hash
from mal_recommender.works import build_works


def _seed_trait(conn, item_id, node):
    label = heuristic_traits(node)
    conn.execute(
        """
        INSERT INTO item_traits (
          canonical_item_id, prompt_version, model_name, source_hash, traits_json, confidence
        )
        VALUES (?, 'traits-v1', 'heuristic-v1', ?, ?, ?)
        ON CONFLICT(canonical_item_id, prompt_version) DO UPDATE SET
          traits_json = excluded.traits_json,
          confidence = excluded.confidence
        """,
        (item_id, source_hash(node), db.dumps(label.model_dump()), label.confidence),
    )


def test_build_works_groups_sequels_and_marks_entrypoint(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    season_1 = {
        "id": 1,
        "title": "Mob Psycho 100",
        "synopsis": "A comedy with psychic friends.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2016-07-01",
    }
    season_2 = {
        "id": 2,
        "title": "Mob Psycho 100 II",
        "synopsis": "The next season.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2019-01-01",
    }
    ova = {
        "id": 3,
        "title": "Mob Psycho 100 OVA",
        "synopsis": "Bonus story.",
        "genres": [{"name": "Comedy"}],
        "media_type": "ova",
        "start_date": "2019-09-01",
    }
    with db.session(path) as conn:
        item_1 = upsert_item(conn, "anime", season_1)
        item_2 = upsert_item(conn, "anime", season_2)
        item_3 = upsert_item(conn, "anime", ova)
        upsert_item_relation(conn, item_1, item_2, "sequel", "anilist")
        upsert_item_relation(conn, item_1, item_3, "side_story", "anilist")
    result = build_works("anime", path)
    with db.session(path) as conn:
        links = conn.execute(
            """
            SELECT i.title, l.role, l.is_entrypoint, l.sequence_index
            FROM item_work_links l
            JOIN canonical_items i ON i.id = l.canonical_item_id
            ORDER BY l.sequence_index
            """
        ).fetchall()

    assert result.works == 1
    assert result.links == 3
    assert links[0]["title"] == "Mob Psycho 100"
    assert links[0]["is_entrypoint"]
    assert links[2]["role"] == "ova"


def test_recommend_surfaces_one_entrypoint_per_work(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: type("S", (), {"sqlite_path": path})())
    season_1 = {
        "id": 10,
        "title": "Cozy Work",
        "synopsis": "Friends tell jokes.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2020-01-01",
    }
    season_2 = {
        "id": 11,
        "title": "Cozy Work II",
        "synopsis": "More friends tell jokes.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2021-01-01",
    }
    other = {
        "id": 12,
        "title": "Another Cozy",
        "synopsis": "A relaxed story.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2022-01-01",
    }
    with db.session(path) as conn:
        item_1 = upsert_item(conn, "anime", season_1)
        item_2 = upsert_item(conn, "anime", season_2)
        item_3 = upsert_item(conn, "anime", other)
        for item_id, node in [(item_1, season_1), (item_2, season_2), (item_3, other)]:
            _seed_trait(conn, item_id, node)
        upsert_item_relation(conn, item_1, item_2, "sequel", "anilist")
    build_works("anime", path)

    _, results = recommend(RecommendationRequest(mood="cozy", content_types=["anime"], limit=10))
    titles = [result.title for result in results]

    assert "Cozy Work" in titles
    assert "Cozy Work II" not in titles
    assert len(titles) == 2
