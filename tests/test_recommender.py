from mal_recommender import db
from mal_recommender.ingest import add_history_event, upsert_canonical_item, upsert_entry, upsert_item, upsert_user
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
    item_id = upsert_item(conn, content_type, node)
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
    return item_id


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


def test_plan_to_watch_boosts_anime_candidate(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: type("S", (), {"sqlite_path": path})())
    with db.session(path) as conn:
        upsert_user(conn, {"id": 1, "name": "tester"})
        _seed_item(conn, "anime", 20, "Regular Cozy", ["Comedy"], "Friends relax and tell jokes.", 12)
        _seed_item(
            conn,
            "anime",
            21,
            "Wishlisted Cozy",
            ["Comedy"],
            "Friends relax and tell jokes.",
            12,
            "plan_to_watch",
        )

    _, results = recommend(RecommendationRequest(user_id=1, mood="cozy", content_types=["anime"], limit=2))

    assert [result.title for result in results] == ["Wishlisted Cozy", "Regular Cozy"]
    assert "on your MAL plan-to-watch list" in results[0].reasons


def test_plan_to_read_boosts_manga_candidate(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: type("S", (), {"sqlite_path": path})())
    with db.session(path) as conn:
        upsert_user(conn, {"id": 1, "name": "tester"})
        _seed_item(conn, "manga", 30, "Regular Manga", ["Comedy"], "Friends relax and tell jokes.", 40)
        _seed_item(
            conn,
            "manga",
            31,
            "Wishlisted Manga",
            ["Comedy"],
            "Friends relax and tell jokes.",
            40,
            "plan_to_read",
        )

    _, results = recommend(RecommendationRequest(user_id=1, mood="cozy", content_types=["manga"], limit=2))

    assert [result.title for result in results] == ["Wishlisted Manga", "Regular Manga"]
    assert "on your MAL plan-to-read list" in results[0].reasons


def test_on_hold_boost_is_weaker_than_plan_to_watch(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: type("S", (), {"sqlite_path": path})())
    with db.session(path) as conn:
        upsert_user(conn, {"id": 1, "name": "tester"})
        _seed_item(conn, "anime", 40, "On Hold Cozy", ["Comedy"], "Friends relax and tell jokes.", 12, "on_hold")
        _seed_item(
            conn,
            "anime",
            41,
            "Plan Cozy",
            ["Comedy"],
            "Friends relax and tell jokes.",
            12,
            "plan_to_watch",
        )

    _, results = recommend(RecommendationRequest(user_id=1, mood="cozy", content_types=["anime"], limit=2))

    assert [result.title for result in results] == ["Plan Cozy", "On Hold Cozy"]
    assert "on your MAL plan-to-watch list" in results[0].reasons
    assert "on your MAL on-hold list" in results[1].reasons


def test_wishlist_boost_does_not_override_much_better_trait_match(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: type("S", (), {"sqlite_path": path})())
    with db.session(path) as conn:
        upsert_user(conn, {"id": 1, "name": "tester"})
        _seed_item(
            conn,
            "anime",
            50,
            "Wishlisted Heavy",
            ["Drama"],
            "War, death, grief, and tragedy shape everyone.",
            12,
            "plan_to_watch",
        )
        _seed_item(conn, "anime", 51, "Comfort Match", ["Comedy"], "Friends relax and tell jokes.", 12)

    _, results = recommend(
        RecommendationRequest(
            user_id=1,
            mode=RecommendationMode.cooldown_after_heavy,
            mood="cozy",
            content_types=["anime"],
            limit=2,
        )
    )

    assert results[0].title == "Comfort Match"


def test_manual_saved_event_is_not_mal_wishlist_intent(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: type("S", (), {"sqlite_path": path})())
    with db.session(path) as conn:
        upsert_user(conn, {"id": 1, "name": "tester"})
        item_id = _seed_item(conn, "anime", 60, "Manual Save", ["Comedy"], "Friends relax and tell jokes.", 12)
        add_history_event(conn, 1, item_id, "anime", "saved", "manual")

    _, results = recommend(RecommendationRequest(user_id=1, mood="cozy", content_types=["anime"], limit=1))

    assert results[0].title == "Manual Save"
    assert all("MAL plan" not in reason and "MAL on-hold" not in reason for reason in results[0].reasons)


def test_seen_status_matches_anilist_duplicate_by_mal_id(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: type("S", (), {"sqlite_path": path})())
    duplicate_node = {
        "id": 7000,
        "idMal": 70,
        "title": "Seen Duplicate",
        "synopsis": "Friends relax and tell jokes.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "num_episodes": 12,
    }
    with db.session(path) as conn:
        upsert_user(conn, {"id": 1, "name": "tester"})
        _seed_item(conn, "anime", 70, "Seen Duplicate", ["Comedy"], "Friends relax and tell jokes.", 12, "completed", 9)
        duplicate_id = upsert_canonical_item(conn, "anilist", "anime", 7000, duplicate_node)
        label = heuristic_traits(duplicate_node)
        conn.execute(
            """
            INSERT INTO item_traits (
              canonical_item_id, prompt_version, model_name, source_hash, traits_json, confidence
            )
            VALUES (?, 'traits-v1', 'heuristic-v1', ?, ?, ?)
            """,
            (duplicate_id, source_hash(duplicate_node), db.dumps(label.model_dump()), label.confidence),
        )
        _seed_item(conn, "anime", 71, "Visible Pick", ["Comedy"], "Friends relax and tell jokes.", 12)

    _, results = recommend(RecommendationRequest(user_id=1, mood="cozy", content_types=["anime"], limit=10))

    assert "Seen Duplicate" not in [result.title for result in results]
    assert "Visible Pick" in [result.title for result in results]


def test_psychological_mood_prefers_tense_over_cozy_pick(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    db.init_db(path)
    monkeypatch.setattr("mal_recommender.db.get_settings", lambda: type("S", (), {"sqlite_path": path})())
    with db.session(path) as conn:
        upsert_user(conn, {"id": 1, "name": "tester"})
        _seed_item(
            conn,
            "anime",
            80,
            "Cozy School Comedy",
            ["Comedy", "Slice of Life"],
            "Friends relax in a school club and tell jokes.",
            12,
        )
        _seed_item(
            conn,
            "anime",
            81,
            "Tense Mystery",
            ["Psychological", "Suspense"],
            "A detective faces a serial killer in a psychological mystery.",
            12,
        )

    _, results = recommend(RecommendationRequest(user_id=1, mood="psychological", content_types=["anime"], limit=2))

    assert results[0].title == "Tense Mystery"
    assert "matches mood: psychological" in results[0].reasons
