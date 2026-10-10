"""Synthetic station-congestion dataset: python -m src.dataset [-> data/congestion.csv].

IMPORTANT: this data is SYNTHETIC. It is generated from the stated assumptions
below, NOT from real TransMilenio measurements. The coefficients are plausible
guesses for academic use; a model trained on this data only learns the rules
written here (plus noise), not real passenger behavior.

Latent congestion score per row:

    score = hour/day effect + TRANSFER + RAIN + EVENT + gaussian noise

and the class is chosen by two thresholds (LOW_MAX, MEDIUM_MAX).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NETWORK_PATH = ROOT / "data" / "network.json"
CSV_PATH = ROOT / "data" / "congestion.csv"

FEATURES = ["station", "line", "hour", "day_type", "is_transfer", "weather", "near_event"]
TARGET = "congestion"
CLASSES = ["low", "medium", "high"]
DAY_TYPES = ["weekday", "saturday", "sunday_holiday"]
DAY_TYPE_PROBS = [0.6, 0.2, 0.2]
RAIN_PROB = 0.25
EVENT_PROB = 0.10
HOURS = range(5, 23)

# Assumed contributions to the latent score (arbitrary units).
PEAK_HOURS = (6, 7, 8, 17, 18, 19)
WEEKDAY_PEAK = 2.0
WEEKDAY_DAYTIME = 0.8  # weekday, 09-16
SATURDAY = 0.2
SUNDAY_HOLIDAY = -0.8
LATE_NIGHT = -1.0  # 21 h onwards
TRANSFER = 0.8
RAIN = 0.6
EVENT = 0.8
NOISE_STD = 0.5
LOW_MAX = 0.5  # score < LOW_MAX -> low
MEDIUM_MAX = 1.8  # LOW_MAX <= score < MEDIUM_MAX -> medium, else high


def station_lines(network_path: Path = NETWORK_PATH) -> dict:
    """Map each station name to the sorted list of lines that serve it."""
    network = json.loads(Path(network_path).read_text(encoding="utf-8"))
    result: dict = {}
    for line, stations in network["lines"].items():
        for station in stations:
            result.setdefault(station, []).append(line)
    return {station: sorted(lines) for station, lines in result.items()}


def _time_effect(hour: int, day_type: str) -> float:
    if day_type == "weekday":
        if hour in PEAK_HOURS:
            effect = WEEKDAY_PEAK
        elif 9 <= hour <= 16:
            effect = WEEKDAY_DAYTIME
        else:
            effect = 0.0
    elif day_type == "saturday":
        effect = SATURDAY
    else:
        effect = SUNDAY_HOLIDAY
    return effect + (LATE_NIGHT if hour >= 21 else 0.0)


def generate_dataset(n: int = 3000, seed: int = 42, network_path: Path = NETWORK_PATH) -> pd.DataFrame:
    """Return n synthetic rows, one per sampled (station, line) pair and context."""
    lines_of = station_lines(network_path)
    pairs = [(s, line) for s, lines in lines_of.items() for line in lines]
    rng = np.random.default_rng(seed)
    picked = rng.integers(0, len(pairs), n)
    df = pd.DataFrame(
        {
            "station": [pairs[i][0] for i in picked],
            "line": [pairs[i][1] for i in picked],
            "hour": rng.choice(list(HOURS), n),
            "day_type": rng.choice(DAY_TYPES, n, p=DAY_TYPE_PROBS),
            "weather": np.where(rng.random(n) < RAIN_PROB, "rain", "dry"),
            "near_event": (rng.random(n) < EVENT_PROB).astype(int),
        }
    )
    df["is_transfer"] = df["station"].map(lambda s: int(len(lines_of[s]) >= 2))
    score = (
        [_time_effect(h, d) for h, d in zip(df["hour"], df["day_type"])]
        + TRANSFER * df["is_transfer"]
        + RAIN * (df["weather"] == "rain")
        + EVENT * df["near_event"]
        + rng.normal(0.0, NOISE_STD, n)
    )
    df[TARGET] = np.select([score < LOW_MAX, score < MEDIUM_MAX], CLASSES[:2], default=CLASSES[2])
    return df[FEATURES + [TARGET]]


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv(CSV_PATH, index=False)
    print(f"Wrote {len(data)} rows to {CSV_PATH.relative_to(ROOT)}")
    print(data[TARGET].value_counts(normalize=True).round(3).to_string())
