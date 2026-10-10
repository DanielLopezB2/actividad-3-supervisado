import json
from pathlib import Path

import pandas as pd
import pytest

from src.dataset import CLASSES, FEATURES, TARGET, generate_dataset

ROOT = Path(__file__).resolve().parent.parent
NETWORK = ROOT / "data" / "network.json"


@pytest.fixture(scope="module")
def df():
    return generate_dataset(n=3000, seed=42)


def test_schema_and_size(df):
    assert len(df) == 3000
    assert list(df.columns) == FEATURES + [TARGET]
    assert df["hour"].between(5, 22).all()
    assert set(df["day_type"]) == {"weekday", "saturday", "sunday_holiday"}
    assert set(df["weather"]) == {"dry", "rain"}
    assert set(df["near_event"]) <= {0, 1}
    assert df.isna().sum().sum() == 0


def test_same_seed_is_deterministic(df):
    pd.testing.assert_frame_equal(df, generate_dataset(n=3000, seed=42))


def test_different_seed_differs(df):
    assert not df.equals(generate_dataset(n=3000, seed=7))


def test_all_classes_present_in_reasonable_proportion(df):
    shares = df[TARGET].value_counts(normalize=True)
    assert set(shares.index) == set(CLASSES)
    assert shares.min() >= 0.15


def test_is_transfer_matches_network(df):
    network = json.loads(NETWORK.read_text(encoding="utf-8"))
    lines_per_station = {}
    for line, stations in network["lines"].items():
        for station in stations:
            lines_per_station.setdefault(station, set()).add(line)
    for row in df.itertuples():
        assert row.line in lines_per_station[row.station]
        assert row.is_transfer == int(len(lines_per_station[row.station]) >= 2)
    assert df["is_transfer"].nunique() == 2
