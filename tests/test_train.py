import joblib
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from src import train
from src.dataset import FEATURES


@pytest.fixture(scope="module")
def split():
    X, y = train.load_data()
    return train.split(X, y)


@pytest.fixture(scope="module")
def fitted(split):
    X_train, _, y_train, _ = split
    return train.train(X_train, y_train)


def test_load_data_returns_features_and_target():
    X, y = train.load_data()
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) > 0


def test_split_is_stratified_80_20(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_test) == pytest.approx(0.2 * (len(X_train) + len(X_test)), abs=1)
    for label in y_train.unique():
        assert (y_train == label).mean() == pytest.approx((y_test == label).mean(), abs=0.01)


def test_tree_beats_baseline_by_clear_margin(split, fitted):
    X_train, X_test, y_train, y_test = split
    model, _ = fitted
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))
    result = train.evaluate(model, X_test, y_test)
    assert result["accuracy"] >= baseline_acc + 0.25
    assert set(result) >= {"accuracy", "report", "confusion_matrix", "labels"}


def test_training_is_deterministic(split, fitted):
    X_train, X_test, y_train, _ = split
    model, params = fitted
    model2, params2 = train.train(X_train, y_train)
    assert params == params2
    assert list(model.predict(X_test)) == list(model2.predict(X_test))


def test_best_params_are_searched_values(fitted):
    _, params = fitted
    assert set(params) == {"max_depth", "min_samples_leaf"}
    assert params["max_depth"] in train.PARAM_GRID["max_depth"]
    assert params["min_samples_leaf"] in train.PARAM_GRID["min_samples_leaf"]


def test_main_saves_model_and_plot(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(train, "MODEL_PATH", tmp_path / "models" / "tree.joblib")
    monkeypatch.setattr(train, "PLOT_PATH", tmp_path / "img" / "tree.png")
    train.main()
    out = capsys.readouterr().out
    for expected in ("Baseline accuracy", "Tree accuracy", "Confusion matrix", "|---"):
        assert expected in out
    assert (tmp_path / "img" / "tree.png").stat().st_size > 0
    assert joblib.load(tmp_path / "models" / "tree.joblib").predict(train.load_data()[0][:1]).shape == (1,)
