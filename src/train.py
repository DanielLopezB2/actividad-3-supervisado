"""Train and evaluate the congestion decision tree: python -m src.train."""
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # noqa: E402

from src.dataset import CSV_PATH, FEATURES, ROOT, TARGET, generate_dataset  # noqa: E402

MODEL_PATH = ROOT / "models" / "congestion_tree.joblib"
PLOT_PATH = ROOT / "docs" / "img" / "arbol-decision.png"
SEED = 42
CATEGORICAL = ["station", "line", "day_type", "weather"]
# Leaves of at least 10 rows keep the rules from memorizing single samples.
PARAM_GRID = {"max_depth": [3, 4, 5, 6, 8], "min_samples_leaf": [10, 30, 60]}


def load_data(path: Path = CSV_PATH):
    """Return (X, y) from the CSV, generating it first if it does not exist."""
    if not Path(path).exists():
        generate_dataset().to_csv(path, index=False)
    df = pd.read_csv(path)
    return df[FEATURES], df[TARGET]


def split(X, y):
    """Stratified 80/20 train/test split."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def make_pipeline() -> Pipeline:
    """One-hot encode categoricals, then a decision tree (no manual encoding needed to predict)."""
    encoder = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return Pipeline([("encode", encoder), ("tree", DecisionTreeClassifier(random_state=SEED))])


def train(X_train, y_train):
    """Pick max_depth / min_samples_leaf by 5-fold CV; return (fitted pipeline, best params)."""
    grid = {f"tree__{name}": values for name, values in PARAM_GRID.items()}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(make_pipeline(), grid, cv=cv, scoring="accuracy").fit(X_train, y_train)
    params = {key.removeprefix("tree__"): value for key, value in search.best_params_.items()}
    return search.best_estimator_, params


def evaluate(model, X_test, y_test) -> dict:
    """Accuracy, classification report and confusion matrix on held-out data."""
    pred = model.predict(X_test)
    labels = list(model.classes_)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "report": classification_report(y_test, pred, labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels),
        "labels": labels,
    }


def main() -> None:
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    model, params = train(X_train, y_train)
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    result = evaluate(model, X_test, y_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(make_pipeline().set_params(**{f"tree__{k}": v for k, v in params.items()}), X_train, y_train, cv=cv)

    print(f"Rows: {len(X)} | train: {len(X_train)} | test: {len(X_test)}")
    print(f"Chosen hyperparameters: {params}")
    print(f"Baseline accuracy (most frequent class): {accuracy_score(y_test, baseline.predict(X_test)):.3f}")
    print(f"Tree accuracy (held-out test): {result['accuracy']:.3f}")
    print(f"5-fold CV accuracy (train split): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print("\nClassification report:\n" + result["report"])
    print(f"Confusion matrix (rows = true, columns = predicted, order = {result['labels']}):")
    print(result["confusion_matrix"])

    tree = model.named_steps["tree"]
    names = list(model.named_steps["encode"].get_feature_names_out())
    ranked = sorted(zip(names, tree.feature_importances_), key=lambda item: -item[1])
    print("\nTop feature importances:")
    for name, importance in ranked[:6]:
        print(f"  {name:<24}{importance:.3f}")
    print("\nDecision rules:\n" + export_text(tree, feature_names=names))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(22, 10))
    plot_tree(tree, feature_names=names, class_names=list(tree.classes_), filled=True, max_depth=3, fontsize=8)
    fig.savefig(PLOT_PATH, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Model saved to {MODEL_PATH}\nTree plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    main()
