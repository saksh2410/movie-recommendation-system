"""Serialization helpers for generated recommendation artifacts."""

import pickle
from pathlib import Path
from typing import Any

from . import config


def save_pickle(obj: Any, path: Path) -> None:
    """Serialize an object to a pickle file, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: Path) -> Any:
    """Load a Python object from a pickle file."""
    with path.open("rb") as file:
        return pickle.load(file)


def save_recommendation_artifacts(movies, similarity) -> None:
    """Persist the processed movie dataframe and similarity matrix."""
    save_pickle(movies, config.MOVIES_ARTIFACT)
    save_pickle(similarity, config.SIMILARITY_ARTIFACT)


def load_recommendation_artifacts():
    """Load the processed movie dataframe and similarity matrix."""
    return (
        load_pickle(config.MOVIES_ARTIFACT),
        load_pickle(config.SIMILARITY_ARTIFACT),
    )
