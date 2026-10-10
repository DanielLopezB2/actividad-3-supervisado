# Station Congestion Classifier for Mass Transit

## Project Overview

This project applies supervised learning to the TransMilenio (Bogotá) network used in the previous activity (4 lines, 20 stations). It trains a decision tree that classifies the congestion level of a station (`low`, `medium`, `high`) from its context: hour, day type, weather, nearby events and whether it is a transfer station.

The system combines:

- A reproducible synthetic dataset.
- A decision tree classifier with human-readable rules.
- Cross-validated hyperparameter selection and baseline comparison.
- A command-line predictor.
- Automated testing.

## Academic Purpose

The project demonstrates the application of artificial intelligence concepts related to:

1. Supervised learning (classification).
2. Decision trees and decision rules.
3. Model evaluation (train/test split, cross-validation, confusion matrix).
4. Interpretable decision-making.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.dataset   # writes data/congestion.csv
python -m src.train     # trains, evaluates, saves model and tree plot
python -m src.predict --station "Calle 100" --hour 7 --day weekday --weather rain --event 0
python -m pytest
```

CLI messages are in Spanish. `--day` accepts `weekday`, `saturday` or `sunday_holiday`; `--weather` accepts `dry` or `rain`; `--hour` is 5 to 22. Station names are matched ignoring case and accents, with suggestions on a miss (exit code 2).

## Data Source

There is no public per-station congestion dataset for this network, so the data is **synthetic**: `src/dataset.py` generates it from stated assumptions, not from real TransMilenio measurements. Stations and lines come from `data/network.json` (the same approximate network as the previous activity).

Each row is a (station, line) pair with `hour`, `day_type`, `is_transfer`, `weather` and `near_event`. The label comes from a latent score: weekday peak hours (6-8 and 17-19) weigh most, transfer stations, rain and nearby events add to it, weekends and late nights subtract, plus gaussian noise (seed 42). The coefficients are named constants at the top of `src/dataset.py`.

The consequence is important: the model can only rediscover the rules written into the generator (plus noise). Its accuracy says nothing about real passenger behavior.

## Model and Evaluation

`DecisionTreeClassifier` inside a scikit-learn `Pipeline` (one-hot encoding of categoricals). Stratified 80/20 split, `max_depth` and `min_samples_leaf` chosen by 5-fold cross-validation on the training split. Results of `python -m src.train` with seed 42 (3000 rows: 1228 low, 935 medium, 837 high):

| Metric | Value |
| --- | --- |
| Baseline accuracy (most frequent class) | 0.410 |
| Tree accuracy (held-out 20%) | 0.787 |
| 5-fold CV accuracy (train split) | 0.759 (+/- 0.019) |
| Chosen hyperparameters | `max_depth=6`, `min_samples_leaf=10` |

The most important features are `hour`, `day_type`, `is_transfer`, `near_event` and `weather`, as expected from the generator. The medium class is the hardest (F1 0.64) because it sits between the other two. The full rule set is printed by `src.train` (via `export_text`) and the top levels are drawn in `docs/img/arbol-decision.png`.

## Project Structure

```text
actividad-3/
├── README.md
├── requirements.txt
├── data/
│   ├── network.json          # stations and lines (approximate estimates)
│   └── congestion.csv        # synthetic dataset (generated)
├── docs/
│   └── img/arbol-decision.png  # tree plot (generated)
├── src/
│   ├── dataset.py            # synthetic data generator
│   ├── train.py              # training, evaluation, rules, plot
│   └── predict.py            # prediction CLI (es)
└── tests/
```

The trained model is saved to `models/congestion_tree.joblib` (git-ignored); `src.predict` trains it on the fly if it is missing.

## Limitations

- The dataset is synthetic; real congestion depends on factors not modeled here (incidents, fleet size, school calendars).
- Rows are generated from hand-picked coefficients and noise, so the reported accuracy is an upper-bound illustration of the method, not a performance claim.
- Station names and lines are one-hot encoded, so the tree may use them in a few spurious splits even though the generator ignores them.
- The test split is a single 600-row sample; differences of a few points between runs of other seeds are expected.
