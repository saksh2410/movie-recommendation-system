# Movie Recommendation System

A content-based movie recommender built with Python. It uses movie details
such as genres, cast, and plot descriptions to find similar films.


## Setup

Make sure the TMDB movie and credits CSV files are in `data/raw/`. They are
available from the [TMDB 5000 dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Use

Get recommendations for a movie:

```bash
python -m src.main recommend "Iron Man"
```

The first run builds the recommendation data automatically. To choose how many
recommendations to show, use `--top-n`:

```bash
python -m src.main recommend "Iron Man" --top-n 10
```

To build the data ahead of time:

```bash
python -m src.main build
```

Run the tests with:

```bash
pytest
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

## Background

The first version was built in 2022 while learning NLP and recommendation-system fundamentals. In 2026, the project was revisited and restructured into separate modules for configuration, loading, preprocessing, recommendation logic, artifact persistence, and command-line execution.