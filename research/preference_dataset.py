import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from research.preference_data import FEATURE_NAMES, create_preference_pair
from research.trajectory_features import extract_trajectory_features


def generate_preference_pairs(candidates, scenario):
    """
    Generate all unique pairwise comparisons between
    candidate trajectories.

    Preferences remain unlabeled.
    """
    feature_records = []

    for candidate in candidates:
        x, y = candidate["path"]
        features = extract_trajectory_features(x, y, scenario)
        feature_records.append({
            "candidate_id": candidate["candidate_id"],
            "features": features,
        })

    pairs = []
    pair_counter = 1

    for candidate_a, candidate_b in combinations(feature_records, 2):
        pair = create_preference_pair(
            candidate_a=candidate_a["candidate_id"],
            features_a=candidate_a["features"],
            candidate_b=candidate_b["candidate_id"],
            features_b=candidate_b["features"],
            preferred_candidate=None,
        )

        pair["comparison_id"] = (
            f"{scenario['scenario_id']}_comparison_{pair_counter:04d}"
        )
        pair["scenario_id"] = scenario["scenario_id"]

        pairs.append(pair)
        pair_counter += 1

    return pairs


def save_preference_dataset_json(pairs, output_path):
    """Save preference pairs to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_pairs = []

    for pair in pairs:
        serializable_pairs.append({
            "comparison_id": pair["comparison_id"],
            "scenario_id": pair["scenario_id"],
            "candidate_a": pair["candidate_a"],
            "candidate_b": pair["candidate_b"],
            "features_a": pair["features_a"].tolist(),
            "features_b": pair["features_b"].tolist(),
            "preferred_candidate": pair["preferred_candidate"],
        })

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable_pairs, file, indent=4)


def save_preference_dataset_csv(pairs, output_path):
    """
    Save preference pairs to CSV.

    Each trajectory feature gets its own column.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["comparison_id", "scenario_id", "candidate_a", "candidate_b"]

    for feature_name in FEATURE_NAMES:
        fieldnames.append(f"a_{feature_name}")

    for feature_name in FEATURE_NAMES:
        fieldnames.append(f"b_{feature_name}")

    fieldnames.append("preferred_candidate")

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for pair in pairs:
            row = {
                "comparison_id": pair["comparison_id"],
                "scenario_id": pair["scenario_id"],
                "candidate_a": pair["candidate_a"],
                "candidate_b": pair["candidate_b"],
                "preferred_candidate": pair["preferred_candidate"],
            }

            for index, feature_name in enumerate(FEATURE_NAMES):
                row[f"a_{feature_name}"] = float(pair["features_a"][index])
                row[f"b_{feature_name}"] = float(pair["features_b"][index])

            writer.writerow(row)