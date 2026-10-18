"""End-to-end build pipeline for the recommendation system."""

from .artifacts import save_recommendation_artifacts
from .load_data import load_credits, load_movies
from .preprocess import (
    clean_dataframe,
    generate_tags,
    merge_movies_credits,
    stem_tags,
)
from .recommender import (
    calculate_cosine_similarity,
    compute_tfidf,
    create_count_matrix,
)


def build_pipeline():
    """Build processed movie features and the pairwise similarity matrix."""
    movies = load_movies()
    credits = load_credits()

    merged = merge_movies_credits(movies, credits)
    cleaned = clean_dataframe(merged)
    tagged = generate_tags(cleaned)
    tagged = stem_tags(tagged)

    count_matrix = create_count_matrix(tagged["tags"].tolist())
    tfidf_matrix = compute_tfidf(count_matrix)
    similarity = calculate_cosine_similarity(tfidf_matrix)

    return tagged, similarity


def build_and_save():
    """Build the pipeline and persist generated artifacts."""
    movies, similarity = build_pipeline()
    save_recommendation_artifacts(movies, similarity)
    return movies, similarity
