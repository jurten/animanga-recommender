from mal_recommender import db
from mal_recommender.anilist import normalize_anilist_media
from mal_recommender.ingest import add_history_event, canonical_id_for_source, upsert_entry, upsert_item, upsert_user
from mal_recommender.traits import heuristic_traits
from mal_recommender.vectors import embedding_text, input_hash


def test_mal_item_writes_raw_canonical_and_source_link(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    node = {
        "id": 101,
        "title": "Linked Show",
        "synopsis": "A warm comedy.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "num_episodes": 12,
    }

    with db.session(path) as conn:
        canonical_item_id = upsert_item(conn, "anime", node)
        source_record = conn.execute("SELECT * FROM source_records WHERE source = 'mal'").fetchone()
        canonical = conn.execute("SELECT * FROM canonical_items WHERE id = ?", (canonical_item_id,)).fetchone()
        link = conn.execute("SELECT * FROM item_source_links WHERE source = 'mal'").fetchone()

    assert source_record["source_item_id"] == "101"
    assert canonical["title"] == "Linked Show"
    assert link["canonical_item_id"] == canonical_item_id


def test_mal_list_import_creates_source_agnostic_event(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    node = {
        "id": 102,
        "title": "Finished Show",
        "synopsis": "A bright comedy.",
        "genres": [{"name": "Comedy"}],
        "media_type": "tv",
        "num_episodes": 12,
    }
    edge = {
        "node": node,
        "list_status": {"status": "completed", "score": 9, "num_episodes_watched": 12},
    }

    with db.session(path) as conn:
        user_id = upsert_user(conn, {"id": 1, "name": "tester"})
        upsert_item(conn, "anime", node)
        upsert_entry(conn, user_id, "anime", edge)
        event = conn.execute("SELECT * FROM user_item_events").fetchone()

    assert event["event_type"] == "completed"
    assert event["source"] == "mal"
    assert event["score"] == 9


def test_manual_history_event_is_source_agnostic(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    node = {"id": 103, "title": "Saved Show", "media_type": "movie", "genres": []}

    with db.session(path) as conn:
        user_id = upsert_user(conn, {"id": 1, "name": "tester"})
        canonical_item_id = upsert_item(conn, "anime", node)
        event_id = add_history_event(conn, user_id, canonical_item_id, "anime", "saved", "manual")
        event = conn.execute("SELECT * FROM user_item_events WHERE id = ?", (event_id,)).fetchone()

    assert event["event_type"] == "saved"
    assert event["source"] == "manual"


def test_anilist_normalization_can_link_by_mal_id(tmp_path):
    path = tmp_path / "test.db"
    db.init_db(path)
    media = {
        "id": 9001,
        "idMal": 104,
        "type": "ANIME",
        "format": "TV",
        "status": "FINISHED",
        "title": {"romaji": "Romaji", "english": "English", "native": "Native"},
        "synonyms": ["Alt"],
        "description": "Description",
        "genres": ["Drama"],
        "tags": [{"name": "Coming of Age", "rank": 80, "isMediaSpoiler": False}],
        "episodes": 24,
        "chapters": None,
        "volumes": None,
        "averageScore": 82,
        "popularity": 5000,
        "source": "MANGA",
        "updatedAt": 123456,
    }
    normalized = normalize_anilist_media(media)

    with db.session(path) as conn:
        upsert_item(conn, "anime", {"id": 104, "title": "MAL Title", "media_type": "tv", "genres": []})
        linked_id = canonical_id_for_source(conn, "mal", "anime", normalized["idMal"])

    assert normalized["title"] == "English"
    assert normalized["themes"] == ["Coming of Age"]
    assert linked_id is not None


def test_embedding_text_hash_changes_with_traits():
    item = {
        "content_type": "anime",
        "title": "Quiet Days",
        "synopsis": "Friends relax.",
        "media_type": "tv",
        "num_units": 12,
        "payload_json": db.dumps({"genres": [{"name": "Slice of Life"}]}),
    }
    label = heuristic_traits({"title": "Quiet Days", "synopsis": "Friends relax.", "genres": [{"name": "Slice of Life"}]})
    text = embedding_text(item, label.model_dump())

    assert "Quiet Days" in text
    assert "Slice of Life" in text
    changed = text + "\ncomfort:high"
    assert input_hash(text) != input_hash(changed)
