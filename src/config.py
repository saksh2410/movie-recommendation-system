"""Central configuration for the movie recommendation system."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
MOVIES = RAW_DATA_PATH / "tmdb_5000_movies.csv"
CREDITS = RAW_DATA_PATH / "tmdb_5000_credits.csv"

ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
MOVIES_ARTIFACT = ARTIFACTS_PATH / "movies.pkl"
SIMILARITY_ARTIFACT = ARTIFACTS_PATH / "similarity.pkl"

KEEP_COLS = [
    "title",
    "genres",
    "keywords",
    "overview",
    "tagline",
    "cast",
    "crew",
]
CLEANUP_COLS = ["genres", "keywords", "cast"]

TOP_N = 5
