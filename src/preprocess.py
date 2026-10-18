"""Preprocessing and feature-engineering utilities."""

import ast

import pandas as pd
from nltk.stem import PorterStemmer

from . import config


def merge_movies_credits(
    df_movies: pd.DataFrame, df_credits: pd.DataFrame
) -> pd.DataFrame:
    """Merge movie metadata and credits on title."""
    return pd.merge(df_movies, df_credits, on="title")


def _extract_names(value: str) -> list[str]:
    """Extract and normalize names from a serialized list of dictionaries."""
    items = ast.literal_eval(value)
    return [item["name"].replace(" ", "") for item in items]


def _extract_director(crew: str) -> str:
    """Extract and normalize the director name from serialized crew metadata."""
    for crew_member in ast.literal_eval(crew):
        if crew_member.get("job") == "Director":
            return crew_member["name"].replace(" ", "")
    return ""


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant fields, remove invalid rows, and normalize metadata."""
    cleaned = df.loc[:, config.KEEP_COLS].copy()
    cleaned = cleaned.drop_duplicates().dropna()

    for column in config.CLEANUP_COLS:
        cleaned[column] = cleaned[column].apply(_extract_names)

    cleaned["director"] = cleaned["crew"].apply(_extract_director)
    cleaned = cleaned.drop(columns="crew")
    return cleaned


def generate_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Combine selected movie attributes into one text representation."""
    tags = (
        df["overview"].astype(str)
        + " "
        + df["tagline"].astype(str)
        + " "
        + df["genres"].apply(" ".join)
        + " "
        + df["keywords"].apply(" ".join)
        + " "
        + df["cast"].apply(" ".join)
        + " "
        + df["director"].astype(str)
    )

    return pd.DataFrame({"title": df["title"].values, "tags": tags.values})


def stem_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Porter stemming to the generated movie tags."""
    stemmer = PorterStemmer()
    stemmed = df.copy()

    stemmed["tags"] = stemmed["tags"].apply(
        lambda text: " ".join(stemmer.stem(word) for word in text.split())
    )
    return stemmed
