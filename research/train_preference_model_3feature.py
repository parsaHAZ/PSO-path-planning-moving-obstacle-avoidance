from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = ROOT_DIR / "data" / "preference" / "preference_pairs_annotated_comfort_fixed.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "preference_comfort_fixed_3feature"


FEATURES = [
    "safety",
    "efficiency",
    "comfort",
]


def load_data(path: Path) -> pd.DataFrame:
    print("=" * 72)
    print("LOAD ANNOTATED PREFERENCE DATA")
    print("=" * 72)
    print(f"Input: {path}")

    df = pd.read_csv(path)

    required = [
        "pair_id",
        "scenario_id",
        "drone",
        "graph_index",
        "time",
        "ego_node",
        "candidate_a_safety",
        "candidate_a_efficiency",
        "candidate_a_comfort",
        "candidate_a_path_length",
        "candidate_b_safety",
        "candidate_b_efficiency",
        "candidate_b_comfort",
        "candidate_b_path_length",
        "preferred",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    df["preferred"] = (
        df["preferred"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid_labels = {"A", "B", "TIE"}

    invalid = ~df["preferred"].isin(valid_labels)

    if invalid.any():
        raise ValueError(
            f"Found {invalid.sum()} invalid labels: "
            f"{sorted(df.loc[invalid, 'preferred'].unique())}"
        )

    ties = (df["preferred"] == "TIE").sum()

    if ties:
        print(f"WARNING: {ties} tie labels found.")
        print("This model currently uses decisive A/B comparisons only.")

    df = df[df["preferred"].isin(["A", "B"])].copy()

    if len(df) == 0:
        raise ValueError("No decisive A/B comparisons found.")

    print(f"Rows loaded: {len(df)}")
    print(f"A wins: {(df['preferred'] == 'A').sum()}")
    print(f"B wins: {(df['preferred'] == 'B').sum()}")
    print(f"Ties ignored: {ties}")
    print()

    return df


def build_pairwise_dataset(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert candidate A/B comparisons into pairwise feature differences.

    X = candidate_A_features - candidate_B_features

    y:
        1 -> A preferred
        0 -> B preferred

    The model therefore learns which direction of feature difference
    corresponds to human preference.

    Important:
    candidate mode is NOT used as a feature.
    """

    a_cols = [f"candidate_a_{f}" for f in FEATURES]
    b_cols = [f"candidate_b_{f}" for f in FEATURES]

    A = df[a_cols].astype(float).to_numpy()
    B = df[b_cols].astype(float).to_numpy()

    X = A - B

    y = (df["preferred"].to_numpy() == "A").astype(int)

    groups = df["scenario_id"].astype(str).to_numpy()

    pair_features = pd.DataFrame(
        X,
        columns=[f"delta_{f}" for f in FEATURES],
    )

    pair_features["preferred"] = y
    pair_features["scenario_id"] = groups
    pair_features["pair_id"] = df["pair_id"].to_numpy()

    return X, y, groups, pair_features


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_fold(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    result = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "n_test": int(len(y_test)),
    }

    if len(np.unique(y_test)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(y_test, probabilities)
        )
    else:
        result["roc_auc"] = None

    return result


def grouped_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> tuple[list[dict], np.ndarray]:
    unique_groups = np.unique(groups)

    actual_splits = min(n_splits, len(unique_groups))

    if actual_splits < 2:
        raise ValueError(
            "Need at least 2 unique scenarios for grouped CV."
        )

    print("=" * 72)
    print("GROUPED CROSS-VALIDATION")
    print("=" * 72)
    print(f"Unique scenarios: {len(unique_groups)}")
    print(f"CV folds: {actual_splits}")
    print()

    splitter = GroupKFold(n_splits=actual_splits)

    fold_results = []
    oof_predictions = np.full(len(y), np.nan)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups),
        start=1,
    ):
        model = make_model()

        model.fit(X[train_idx], y[train_idx])

        probabilities = model.predict_proba(X[test_idx])[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        oof_predictions[test_idx] = probabilities

        metrics = evaluate_fold(
            model,
            X[test_idx],
            y[test_idx],
        )

        test_groups = sorted(set(groups[test_idx]))

        print(
            f"Fold {fold:02d} | "
            f"train={len(train_idx):3d} | "
            f"test={len(test_idx):3d} | "
            f"scenarios={test_groups}"
        )

        print(
            f"         Accuracy={metrics['accuracy']:.4f} | "
            f"Balanced Acc={metrics['balanced_accuracy']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']}"
        )

        fold_results.append(
            {
                "fold": fold,
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "test_scenarios": test_groups,
                **metrics,
            }
        )

    return fold_results, oof_predictions


def print_cv_summary(fold_results: list[dict]) -> None:
    print()
    print("=" * 72)
    print("CV SUMMARY")
    print("=" * 72)

    accuracy = [
        x["accuracy"]
        for x in fold_results
    ]

    balanced_accuracy = [
        x["balanced_accuracy"]
        for x in fold_results
    ]

    auc = [
        x["roc_auc"]
        for x in fold_results
        if x["roc_auc"] is not None
    ]

    print(
        f"Accuracy:          "
        f"{np.mean(accuracy):.4f} ± {np.std(accuracy):.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{np.mean(balanced_accuracy):.4f} ± "
        f"{np.std(balanced_accuracy):.4f}"
    )

    if auc:
        print(
            f"ROC-AUC:           "
            f"{np.mean(auc):.4f} ± {np.std(auc):.4f}"
        )

    print()


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
) -> Pipeline:
    print("=" * 72)
    print("TRAIN FINAL PREFERENCE MODEL")
    print("=" * 72)

    model = make_model()
    model.fit(X, y)

    print("Final model trained on all decisive comparisons.")
    print()

    return model


def print_model_weights(model: Pipeline) -> dict:
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]

    coefficients = classifier.coef_[0]

    print("=" * 72)
    print("LEARNED PREFERENCE WEIGHTS")
    print("=" * 72)

    weights = {}

    for feature, coefficient in zip(FEATURES, coefficients):
        weights[feature] = float(coefficient)

        print(
            f"{feature:15s}: "
            f"{coefficient:+.6f}"
        )

    print()
    print(
        "Interpretation:"
    )
    print(
        "Positive weight -> larger A-B difference makes A more likely "
        "to be preferred."
    )
    print(
        "Negative weight -> smaller A-B difference makes A more likely "
        "to be preferred."
    )
    print()

    return weights


def save_oof_predictions(
    df: pd.DataFrame,
    oof_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    result = df[
        [
            "pair_id",
            "scenario_id",
            "drone",
            "graph_index",
            "time",
            "ego_node",
            "candidate_a_mode",
            "candidate_b_mode",
            "preferred",
        ]
    ].copy()

    result["actual"] = (
        result["preferred"] == "A"
    ).astype(int)

    result["predicted_probability_A"] = oof_probabilities

    result["predicted"] = np.where(
        result["predicted_probability_A"] >= 0.5,
        "A",
        "B",
    )

    result["correct"] = (
        result["predicted"] == result["preferred"]
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved OOF predictions: {output_path}")


def save_model(
    model: Pipeline,
    feature_names: list[str],
    cv_results: list[dict],
    weights: dict,
    output_path: Path,
) -> None:
    artifact = {
        "model": model,
        "features": feature_names,
        "model_type": "pairwise_logistic_bradley_terry",
        "training_seed": 42,
        "cv_results": cv_results,
        "weights": weights,
    }

    joblib.dump(
        artifact,
        output_path,
    )

    print(f"Saved model: {output_path}")


def save_metrics(
    df: pd.DataFrame,
    cv_results: list[dict],
    weights: dict,
    output_path: Path,
) -> None:
    metrics = {
        "model_type": "pairwise_logistic_bradley_terry",
        "n_rows_total": int(len(df)),
        "n_scenarios": int(df["scenario_id"].nunique()),
        "n_A_wins": int((df["preferred"] == "A").sum()),
        "n_B_wins": int((df["preferred"] == "B").sum()),
        "features": FEATURES,
        "weights": weights,
        "cv": cv_results,
        "cv_mean_accuracy": float(
            np.mean([x["accuracy"] for x in cv_results])
        ),
        "cv_mean_balanced_accuracy": float(
            np.mean(
                [x["balanced_accuracy"] for x in cv_results]
            )
        ),
        "cv_mean_roc_auc": (
            float(
                np.mean(
                    [
                        x["roc_auc"]
                        for x in cv_results
                        if x["roc_auc"] is not None
                    ]
                )
            )
            if any(x["roc_auc"] is not None for x in cv_results)
            else None
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(f"Saved metrics: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train pairwise human preference model."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data(input_path)

    X, y, groups, pair_features = build_pairwise_dataset(df)

    print("=" * 72)
    print("PAIRWISE DATASET")
    print("=" * 72)
    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Feature differences: A - B")
    print()

    print("Features:")
    for feature in FEATURES:
        print(f"  delta_{feature}")

    print()

    cv_results, oof_probabilities = grouped_cross_validation(
        X,
        y,
        groups,
        args.folds,
    )

    print_cv_summary(cv_results)

    final_model = train_final_model(
        X,
        y,
    )

    weights = print_model_weights(
        final_model,
    )

    model_path = (
        output_dir
        / "preference_model.joblib"
    )

    predictions_path = (
        output_dir
        / "preference_predictions.csv"
    )

    metrics_path = (
        output_dir
        / "preference_metrics.json"
    )

    save_model(
        final_model,
        FEATURES,
        cv_results,
        weights,
        model_path,
    )

    save_oof_predictions(
        df,
        oof_probabilities,
        predictions_path,
    )

    save_metrics(
        df,
        cv_results,
        weights,
        metrics_path,
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Model:       {model_path}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics:     {metrics_path}")
    print()


if __name__ == "__main__":
    main()