from mal_recommender.traits import heuristic_traits


def test_heuristic_traits_detects_cooldown_fit():
    label = heuristic_traits(
        {
            "title": "Quiet Days",
            "synopsis": "Friends relax and tell jokes.",
            "genres": [{"name": "Comedy"}, {"name": "Slice of Life"}],
            "media_type": "tv",
            "num_episodes": 12,
        }
    )

    assert "cozy" in label.moods
    assert label.depth == "shallow"
    assert label.cooldown_fit == "high"
    assert label.filler_risk == "low"
