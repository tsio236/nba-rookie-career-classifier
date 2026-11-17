"""Train the rookie 5-year-retention classifier and save it for the API."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

RANDOM_STATE = 50
TARGET = "TARGET_5Yrs"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "nba_logreg.csv"  # confidential
THRESHOLD = 0.5


def build_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale", MinMaxScaler()),
        ("clf", SVC(kernel="rbf"))
    ])


def make_selector(min_precision):
    """Build a refit rule: best recall among configuration with enough precision."""
    def select_best(cv_results):
        recall = np.asarray(cv_results["mean_test_recall"])
        precision = np.asarray(cv_results["mean_test_precision"])
        eligible = precision >= min_precision
        if not eligible.any():
            raise ValueError("No configuration beats the base-rate prediction.")
        return int(np.argmax(np.where(eligible, recall, -np.inf)))

    return select_best


def main(data_path):
    df = pd.read_csv(data_path)
    y = df[TARGET].to_numpy()
    X = df.drop(columns=[TARGET, "Name"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE, stratify=y
    )

    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            "clf__C": [1, 10, 100],
            "clf__gamma": ["scale", 0.01, 0.1],
            "clf__class_weight": [None, "balanced"]
        },
        scoring={"recall": "recall", "precision": "precision", "accuracy": "accuracy"},
        refit=make_selector(min_precision=y_train.mean() + 0.05),
        cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE),
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)

    eval_model = clone(search.best_estimator_).set_params(clf__probability=True)
    eval_model.fit(X_train, y_train)
    y_pred = (eval_model.predict_proba(X_test)[:, 1] >= THRESHOLD).astype(int)
    print(f"\nTest set (threshold {THRESHOLD}):")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Final model: best configuration refitted on all the data
    final_model = clone(search.best_estimator_).set_params(clf__probability=True)
    final_model.fit(X, y)
    with open(BASE_DIR / "model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(BASE_DIR / "model_columns.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
    print(f"\nSaved model.pkl and model_columns.pkl in {BASE_DIR}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH)
