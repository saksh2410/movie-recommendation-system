# Movie Recommendation System

A content-based movie recommendation system built with NLP-style feature engineering, a custom TF-IDF implementation, and cosine similarity.

The project uses the TMDB 5000 Movie Metadata dataset to represent each movie using its overview, tagline, genres, keywords, cast, and director. It then ranks movies by cosine similarity and returns the top-N titles most similar to a selected movie.

Originally built in 2022 as a learning project and refactored in 2026 to improve modularity, reproducibility, testing, and code organization.

## How it works

1. Load movie metadata and credits.
2. Merge and clean the datasets.
3. Keep the content features used for recommendations:
   - overview
   - tagline
   - genres
   - keywords
   - cast
   - director
4. Normalize multi-word entities such as actor and director names so that `Tom Cruise` becomes `TomCruise`.
5. Combine the selected fields into one text representation per movie.
6. Apply Porter stemming.
7. Build a term-count matrix with `CountVectorizer`.
8. Compute TF-IDF with a custom NumPy implementation.
9. Calculate pairwise cosine similarity.
10. Return the top-N most similar movies.

The default number of recommendations is 5 and can be changed in `src/config.py` or supplied through the CLI.

## Project structure

```text
movieRecommendationSystem/
├── data/
│   └── raw/
│       ├── tmdb_5000_movies.csv
│       └── tmdb_5000_credits.csv
├── src/
│   ├── __init__.py
│   ├── artifacts.py
│   ├── cli.py
│   ├── config.py
│   ├── load_data.py
│   ├── pipeline.py
│   ├── preprocess.py
│   ├── recommender.py
│   └── main.ipynb
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/saksh2410/movieRecommendationSystem.git
cd movieRecommendationSystem

python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The project expects the TMDB movie and credits CSV files under `data/raw/`. The dataset used is available on Kaggle:

https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

## Usage

### Build the recommendation artifacts

```bash
python -m src.cli build
```

This runs the full preprocessing and recommendation pipeline and stores the processed movie dataframe and cosine-similarity matrix under `artifacts/`.

### Get recommendations

```bash
python -m src.cli recommend "Iron Man"
```

Specify a different number of results with:

```bash
python -m src.cli recommend "Iron Man" --top-n 10
```

Example Python usage:

```python
from src.pipeline import build_pipeline
from src.recommender import get_recommendations

movies, similarity = build_pipeline()

recommendations = get_recommendations(
    similarity_matrix=similarity,
    titles=movies["title"].tolist(),
    target_title="Iron Man",
    top_n=5,
)

for title, score in recommendations:
    print(f"{title}: {score:.3f}")
```

## Design choices

### Content-based recommendations

The recommender uses movie metadata rather than user-rating history. This makes it possible to generate recommendations without collaborative-filtering data.

### Entity normalization

Actor and director names have spaces removed before vectorization:

```text
Tom Cruise  -> TomCruise
Tom Holland -> TomHolland
```

Without this step, the shared token `Tom` could create artificial similarity between otherwise unrelated people.

### Custom TF-IDF

`CountVectorizer` is used only to construct the term-count matrix. TF-IDF is then calculated directly with NumPy using smoothed inverse document frequency.

The custom implementation was intentional: the original purpose of the project was to understand the mechanics behind text vectorization rather than treating TF-IDF entirely as a library call.

### Artifact persistence

The processed movie dataframe and cosine-similarity matrix can be serialized to `artifacts/`, allowing recommendations to be served without rebuilding the pipeline each time.

## Testing

Run:

```bash
pytest
```

The tests cover core preprocessing behavior, entity normalization, TF-IDF output, recommendation ranking, and artifact persistence.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- NLTK
- matplotlib
- Jupyter
- pytest

## Limitations

This is a learning-focused content-based recommender rather than a production recommendation service.

- It does not use user ratings or behavioral data.
- It does not implement collaborative filtering.
- Recommendation quality has not been evaluated against a labelled relevance dataset.
- Results depend on the selected metadata fields.
- Pairwise similarity requires O(n²) storage as the catalogue grows.
- There is no deployed API or web interface.

## Possible extensions

- Add an offline relevance-evaluation dataset and ranking metrics.
- Compare the custom TF-IDF implementation against scikit-learn's `TfidfVectorizer`.
- Explore embedding-based semantic representations.
- Expose the recommender through a lightweight API or UI.
- Replace the full similarity matrix with approximate nearest-neighbour search for larger catalogues.

## Background

The first version was built in 2022 while learning NLP and recommendation-system fundamentals. In 2026, the project was revisited and restructured into separate modules for configuration, loading, preprocessing, recommendation logic, artifact persistence, and command-line execution.
