"""Vectorization, similarity, and recommendation functions."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def create_count_matrix(corpus: list[str]) -> np.ndarray:
    """Create a dense term-count matrix from a text corpus."""
    vectorizer = CountVectorizer(stop_words="english")
    return vectorizer.fit_transform(corpus).toarray()


def compute_tfidf(matrix: np.ndarray) -> np.ndarray:
    """Compute a TF-IDF matrix using a custom NumPy implementation.

    Term frequency is normalized by the total term count in each document.
    IDF uses the common smoothed formulation:

        log((1 + n_documents) / (1 + document_frequency)) + 1
    """
    counts = matrix.astype(float)
    row_totals = counts.sum(axis=1, keepdims=True)
    tf = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals != 0,
    )

    n_documents = counts.shape[0]
    document_frequency = (counts > 0).sum(axis=0)
    idf = np.log((1 + n_documents) / (1 + document_frequency)) + 1

    return tf * idf


def calculate_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """Calculate pairwise cosine similarity between matrix rows."""
    return cosine_similarity(matrix)


def get_recommendations(
    similarity_matrix: np.ndarray,
    titles: list[str],
    target_title: str,
    top_n: int = config.TOP_N,
) -> list[tuple[str, float]]:
    """Return the top-N titles most similar to the requested movie."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    try:
        target_index = titles.index(target_title)
    except ValueError as exc:
        raise ValueError(f'Movie "{target_title}" is not in the catalogue') from exc

    similarity_scores = list(enumerate(similarity_matrix[target_index]))
    similarity_scores.sort(key=lambda item: item[1], reverse=True)

    return [
        (titles[index], float(score))
        for index, score in similarity_scores
        if index != target_index
    ][:top_n]
