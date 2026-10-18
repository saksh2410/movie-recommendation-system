"""Data-loading utilities."""

from pathlib import Path

import pandas as pd

from . import config


def load_movies(path: Path = config.MOVIES) -> pd.DataFrame:
    """Load the TMDB movie metadata CSV."""
    return pd.read_csv(path)


def load_credits(path: Path = config.CREDITS) -> pd.DataFrame:
    """Load the TMDB credits CSV."""
    return pd.read_csv(path)
