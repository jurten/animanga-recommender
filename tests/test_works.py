from mal_recommender import db
from mal_recommender.ingest import upsert_entry, upsert_item, upsert_item_relation, upsert_user
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


def _settings(path):
    return type(
        "S",
        (),
        {
            "sqlite_path": path,
            "database_url": f"sqlite:///{path}",
            "embedding_model": "test-model",
        },
    )()


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


def test_recommend_uses_work_vectors_then_selects_entrypoint(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: _settings(path))
    monkeypatch.setattr("mal_recommender.recommender.get_settings", lambda: _settings(path))
    monkeypatch.setattr("mal_recommender.recommender._encode_query_text", lambda text: [1.0, 0.0])
    season_1 = {
        "id": 20,
        "title": "Vector Work",
        "synopsis": "Friends relax together.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2020-01-01",
    }
    season_2 = {
        "id": 21,
        "title": "Vector Work II",
        "synopsis": "More relaxed friends.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2021-01-01",
    }
    off_target = {
        "id": 22,
        "title": "Off Target Work",
        "synopsis": "A grim war story.",
        "genres": [{"name": "Drama"}],
        "media_type": "tv",
        "start_date": "2022-01-01",
    }
    with db.session(path) as conn:
        item_1 = upsert_item(conn, "anime", season_1)
        item_2 = upsert_item(conn, "anime", season_2)
        item_3 = upsert_item(conn, "anime", off_target)
        for item_id, node in [(item_1, season_1), (item_2, season_2), (item_3, off_target)]:
            _seed_trait(conn, item_id, node)
        upsert_item_relation(conn, item_1, item_2, "sequel", "anilist")
    build_works("anime", path)
    with db.session(path) as conn:
        rows = conn.execute("SELECT id, title FROM canonical_works").fetchall()
        by_title = {row["title"]: row["id"] for row in rows}
        conn.execute(
            """
            INSERT INTO work_vectors (canonical_work_id, model_name, input_hash, embedding_json)
            VALUES (?, 'test-model', 'a', ?), (?, 'test-model', 'b', ?)
            """,
            (by_title["Vector Work"], db.dumps([1.0, 0.0]), by_title["Off Target Work"], db.dumps([0.0, 1.0])),
        )

    _, results = recommend(RecommendationRequest(mood="cozy", content_types=["anime"], limit=1))

    assert [result.title for result in results] == ["Vector Work"]
    assert "matched semantic work vector" in results[0].reasons
    assert "selected work entrypoint" in results[0].reasons


def test_recommend_falls_back_without_work_vectors(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: _settings(path))
    monkeypatch.setattr("mal_recommender.recommender.get_settings", lambda: _settings(path))
    node = {
        "id": 30,
        "title": "Fallback Pick",
        "synopsis": "A cozy comedy.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2020-01-01",
    }
    with db.session(path) as conn:
        item_id = upsert_item(conn, "anime", node)
        _seed_trait(conn, item_id, node)

    _, results = recommend(RecommendationRequest(mood="cozy", content_types=["anime"], limit=1))

    assert [result.title for result in results] == ["Fallback Pick"]


def test_strict_release_filtering_unlocks_next_main_release(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: _settings(path))
    season_1 = {
        "id": 40,
        "title": "Strict Work",
        "synopsis": "Friends tell jokes.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2020-01-01",
        "num_episodes": 12,
    }
    season_2 = {
        "id": 41,
        "title": "Strict Work II",
        "synopsis": "The next main season.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2021-01-01",
        "num_episodes": 12,
    }
    ova = {
        "id": 42,
        "title": "Strict Work OVA",
        "synopsis": "Bonus story.",
        "genres": [{"name": "Comedy"}],
        "media_type": "ova",
        "start_date": "2021-06-01",
        "num_episodes": 1,
    }
    recap = {
        "id": 43,
        "title": "Strict Work Recap",
        "synopsis": "A recap.",
        "genres": [{"name": "Comedy"}],
        "media_type": "special",
        "start_date": "2021-07-01",
        "num_episodes": 1,
    }
    movie = {
        "id": 44,
        "title": "Strict Work The Movie",
        "synopsis": "A related movie.",
        "genres": [{"name": "Comedy"}],
        "media_type": "movie",
        "start_date": "2022-01-01",
        "num_episodes": 1,
    }
    with db.session(path) as conn:
        upsert_user(conn, {"id": 1, "name": "tester"})
        item_1 = upsert_item(conn, "anime", season_1)
        item_2 = upsert_item(conn, "anime", season_2)
        item_3 = upsert_item(conn, "anime", ova)
        item_4 = upsert_item(conn, "anime", recap)
        item_5 = upsert_item(conn, "anime", movie)
        for item_id, node in [
            (item_1, season_1),
            (item_2, season_2),
            (item_3, ova),
            (item_4, recap),
            (item_5, movie),
        ]:
            _seed_trait(conn, item_id, node)
        upsert_item_relation(conn, item_1, item_2, "sequel", "anilist")
        upsert_item_relation(conn, item_1, item_3, "side_story", "anilist")
        upsert_item_relation(conn, item_1, item_4, "side_story", "anilist")
        upsert_item_relation(conn, item_1, item_5, "side_story", "anilist")
    build_works("anime", path)

    with db.session(path) as conn:
        upsert_entry(
            conn,
            1,
            "anime",
            {
                "node": season_2,
                "list_status": {"status": "plan_to_watch", "score": None, "num_episodes_watched": 0},
            },
        )
        upsert_entry(
            conn,
            1,
            "anime",
            {
                "node": ova,
                "list_status": {"status": "plan_to_watch", "score": None, "num_episodes_watched": 0},
            },
        )

    _, unseen_results = recommend(RecommendationRequest(mood="cozy", content_types=["anime"], limit=10))
    assert [result.title for result in unseen_results] == ["Strict Work"]

    with db.session(path) as conn:
        upsert_entry(
            conn,
            1,
            "anime",
            {
                "node": season_1,
                "list_status": {"status": "completed", "score": 9, "num_episodes_watched": 12},
            },
        )

    _, unlocked_results = recommend(RecommendationRequest(mood="cozy", content_types=["anime"], limit=10))
    assert [result.title for result in unlocked_results] == ["Strict Work II"]
    assert "on your MAL plan-to-watch list" in unlocked_results[0].reasons

    _, include_seen_results = recommend(
        RecommendationRequest(mood="cozy", content_types=["anime"], limit=10, include_seen=True)
    )
    assert [result.title for result in include_seen_results] == ["Strict Work II"]


def test_recommend_suppresses_extra_even_as_only_work_entrypoint(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: _settings(path))
    movie_extra = {
        "id": 50,
        "title": "Broad Franchise The Movie",
        "synopsis": "A bonus franchise movie.",
        "genres": [{"name": "Comedy"}],
        "media_type": "movie",
        "start_date": "2020-01-01",
        "num_episodes": 1,
    }
    picture_drama = {
        "id": 51,
        "title": "Broad Franchise Picture Drama",
        "synopsis": "A bonus picture drama.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2020-02-01",
        "num_episodes": 1,
    }
    normal = {
        "id": 52,
        "title": "Normal Cozy",
        "synopsis": "Friends tell jokes.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2021-01-01",
        "num_episodes": 12,
    }
    with db.session(path) as conn:
        movie_id = upsert_item(conn, "anime", movie_extra)
        drama_id = upsert_item(conn, "anime", picture_drama)
        normal_id = upsert_item(conn, "anime", normal)
        for item_id, node in [(movie_id, movie_extra), (drama_id, picture_drama), (normal_id, normal)]:
            _seed_trait(conn, item_id, node)
    build_works("anime", path)

    _, results = recommend(RecommendationRequest(mood="cozy", content_types=["anime"], limit=10))

    assert [result.title for result in results] == ["Normal Cozy"]


def test_recommend_suppresses_standalone_sequel_entrypoint(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: _settings(path))
    sequel = {
        "id": 60,
        "title": "Loose Work Season 2",
        "synopsis": "The next season.",
        "genres": [{"name": "Action"}],
        "media_type": "tv",
        "start_date": "2021-01-01",
        "num_episodes": 12,
    }
    normal = {
        "id": 61,
        "title": "Normal Cozy",
        "synopsis": "Friends tell jokes.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "start_date": "2021-01-01",
        "num_episodes": 12,
    }
    with db.session(path) as conn:
        sequel_id = upsert_item(conn, "anime", sequel)
        normal_id = upsert_item(conn, "anime", normal)
        for item_id, node in [(sequel_id, sequel), (normal_id, normal)]:
            _seed_trait(conn, item_id, node)
    build_works("anime", path)

    _, results = recommend(RecommendationRequest(mood="cozy", content_types=["anime"], limit=10))

    assert [result.title for result in results] == ["Normal Cozy"]
