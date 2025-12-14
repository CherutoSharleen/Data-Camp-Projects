# k-NN Item Recommender (Synthetic)

This mini project demonstrates content-based recommendation using cosine similarity on synthetic item feature vectors.

## What the app does

- Loads a fictional item catalog with numeric features.
- Computes cosine similarity between the selected item and all others.
- Returns top-k most similar items.

## Concepts taught

- Vector representation
- Cosine similarity
- Ranking for recommendation

## Installation

No external dependency is required.

```bash
python --version
```

## Run

```bash
cd mini_recommenders/knn_items
python cli.py
```

## Test

```bash
cd mini_recommenders/knn_items
python -m unittest discover -s tests -p "test_*.py"
```

## Representative output

```text
Top recommendations:
- <item_id>: 0.97
- <item_id>: 0.84
```

## Main components

- `recommender.py`: similarity + ranking logic
- `cli.py`: local interactive runner
- `synthetic_items.json`: fictional dataset
- `tests/test_recommender.py`: correctness tests

## Data note

All catalog entries and vectors are fictional/synthetic.

## Limitations and exercises

- Current vectors are hand-crafted and tiny.
- Add min-max scaling, metadata filtering, and explanation text.

## Disclosure

This is a retrospective educational addition created now. Historical author dates may be used for commit metadata and do not indicate the actual execution date of this work.
