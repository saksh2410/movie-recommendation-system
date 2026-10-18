"""Command-line interface for building artifacts and requesting recommendations."""

import argparse

from . import config
from .artifacts import load_recommendation_artifacts
from .pipeline import build_and_save
from .recommender import get_recommendations


def _build_command(_: argparse.Namespace) -> None:
    movies, _ = build_and_save()
    print(f"Built recommendation artifacts for {len(movies):,} movies.")
    print(f"Artifacts written to: {config.ARTIFACTS_PATH}")


def _recommend_command(args: argparse.Namespace) -> None:
    try:
        movies, similarity = load_recommendation_artifacts()
    except FileNotFoundError:
        print("Artifacts not found; building them first...")
        movies, similarity = build_and_save()

    recommendations = get_recommendations(
        similarity_matrix=similarity,
        titles=movies["title"].tolist(),
        target_title=args.movie,
        top_n=args.top_n,
    )

    print(f'\nTop {args.top_n} movies similar to "{args.movie}":')
    for rank, (title, score) in enumerate(recommendations, start=1):
        print(f"{rank:>2}. {title} ({score:.3f})")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Content-based movie recommender")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build and save artifacts")
    build_parser.set_defaults(func=_build_command)

    recommend_parser = subparsers.add_parser(
        "recommend", help="Recommend movies similar to a title"
    )
    recommend_parser.add_argument("movie", help="Movie title as it appears in the dataset")
    recommend_parser.add_argument(
        "--top-n",
        type=int,
        default=config.TOP_N,
        help=f"Number of recommendations (default: {config.TOP_N})",
    )
    recommend_parser.set_defaults(func=_recommend_command)

    return parser


def main() -> None:
    """Run the command-line interface."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
