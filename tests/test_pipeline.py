"""Tests for the core recommendation pipeline."""

import numpy as np
import pandas as pd

from src.artifacts import load_pickle, save_pickle
from src.preprocess import clean_dataframe, generate_tags
from src.recommender import compute_tfidf, get_recommendations


def _sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "title": ["Movie A"],
            "genres": ['[{"id": 1, "name": "Science Fiction"}]'],
            "keywords": ['[{"id": 2, "name": "space travel"}]'],
            "overview": ["A journey through space"],
            "tagline": ["Beyond the stars"],
            "cast": [
                '[{"cast_id": 1, "name": "Tom Cruise"}, '
                '{"cast_id": 2, "name": "Tom Holland"}]'
            ],
            "crew": ['[{"job": "Director", "name": "Jane Doe"}]'],
        }
    )


def test_clean_dataframe_normalizes_entities() -> None:
    cleaned = clean_dataframe(_sample_dataframe())

    assert cleaned.loc[0, "genres"] == ["ScienceFiction"]
    assert cleaned.loc[0, "keywords"] == ["spacetravel"]
    assert cleaned.loc[0, "cast"] == ["TomCruise", "TomHolland"]
    assert cleaned.loc[0, "director"] == "JaneDoe"


def test_generate_tags_keeps_entities_as_tokens() -> None:
    cleaned = clean_dataframe(_sample_dataframe())
    tagged = generate_tags(cleaned)
    tags = tagged.loc[0, "tags"]

    assert "ScienceFiction" in tags
    assert "TomCruise" in tags
    assert "TomHolland" in tags
    assert "T o m" not in tags


def test_compute_tfidf_returns_nonnegative_finite_values() -> None:
    counts = np.array([[2, 0, 1], [1, 1, 0]], dtype=int)
    tfidf = compute_tfidf(counts)

    assert tfidf.shape == counts.shape
    assert np.isfinite(tfidf).all()
    assert (tfidf >= 0).all()


def test_recommendations_exclude_query_movie() -> None:
    similarity = np.array(
        [
            [1.0, 0.9, 0.2],
            [0.9, 1.0, 0.3],
            [0.2, 0.3, 1.0],
        ]
    )
    titles = ["A", "B", "C"]

    recommendations = get_recommendations(similarity, titles, "A", top_n=2)

    assert recommendations == [("B", 0.9), ("C", 0.2)]


def test_pickle_round_trip(tmp_path) -> None:
    path = tmp_path / "artifact.pkl"
    value = {"movies": ["A", "B"]}

    save_pickle(value, path)

    assert load_pickle(path) == value
