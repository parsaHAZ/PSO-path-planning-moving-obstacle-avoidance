"""
Regenerate trajectory feature vectors for the existing human preference
annotations without changing the human preference labels.

The original annotated JSON is preserved.
A new JSON file is written with features calculated using the current
trajectory feature extraction pipeline.
"""

import json
from pathlib import Path

from data.scenarios.scenario_0001 import create_scenario
from pso import PSOPathPlanner
from research.trajectory_features import extract_trajectory_features
from research.preference_data import FEATURE_NAMES


ROOT = Path(__file__).resolve().parent.parent

ANNOTATION_FILE = (
    ROOT / "data" / "preference" / "scenario_0001_annotated.json"
)

OUTPUT_FILE = (
    ROOT / "data" / "preference" / "scenario_0001_preferences_updated.json"
)


def generate_candidates(scenario):
    """Generate the same five PSO candidates used for the pilot data."""

    planner = PSOPathPlanner(
        start=scenario["start"],
        end=scenario["end"],
        n_obstacles=len(scenario["obstacles"]),
        max_iterations=50,
        bounds_x=scenario["bounds_x"],
        bounds_y=scenario["bounds_y"],
        scenario=scenario,
    )

    candidates = planner.generate_candidates(n_candidates=5)

    return candidates


def normalize_candidate(candidate, index):
    """
    Convert a generated PSO candidate into a common representation.

    Supports the candidate dictionary returned by PSOPathPlanner.
    """

    candidate_id = candidate.get(
        "candidate_id",
        f"candidate_{index + 1:02d}",
    )

    path = candidate["path"]

    return candidate_id, path


def main():
    print("=" * 80)
    print("REGENERATING PREFERENCE FEATURE DATA")
    print("=" * 80)

    # ------------------------------------------------------------
    # Load existing human annotations
    # ------------------------------------------------------------
    print("\nLoading human annotations...")

    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if not isinstance(annotations, list):
        raise TypeError("Annotation file must contain a JSON list.")

    print(f"Loaded comparisons: {len(annotations)}")

    # ------------------------------------------------------------
    # Create baseline scenario
    # ------------------------------------------------------------
    scenario = create_scenario(seed=42)

    print(f"Scenario: {scenario['scenario_id']}")
    print(f"Seed: {scenario['seed']}")

    # ------------------------------------------------------------
    # Generate candidates
    # ------------------------------------------------------------
    print("\nGenerating PSO candidates...")

    candidates = generate_candidates(scenario)

    if len(candidates) != 5:
        raise RuntimeError(
            f"Expected 5 candidates, got {len(candidates)}."
        )

    candidate_features = {}

    for index, candidate in enumerate(candidates):
        candidate_id, path = normalize_candidate(candidate, index)

        x, y = path

        features = extract_trajectory_features(
            x,
            y,
            scenario,
        )

        # Convert numpy/scalar values into ordinary Python floats.
        feature_dict = {
            name: float(features[name])
            for name in FEATURE_NAMES
        }

        candidate_features[candidate_id] = feature_dict

        print(f"\n{candidate_id}")

        for name in FEATURE_NAMES:
            print(f"  {name}: {feature_dict[name]:.8f}")

    # ------------------------------------------------------------
    # Validate candidate IDs referenced by annotations
    # ------------------------------------------------------------
    annotation_candidate_ids = set()

    for annotation in annotations:
        annotation_candidate_ids.add(annotation["candidate_a"])
        annotation_candidate_ids.add(annotation["candidate_b"])

    missing = annotation_candidate_ids - set(candidate_features)

    if missing:
        raise RuntimeError(
            "The annotation file references candidates that were not "
            f"generated: {sorted(missing)}"
        )

    # ------------------------------------------------------------
    # Rebuild records
    # ------------------------------------------------------------
    updated_records = []

    for annotation in annotations:
        candidate_a = annotation["candidate_a"]
        candidate_b = annotation["candidate_b"]

        features_a = candidate_features[candidate_a]
        features_b = candidate_features[candidate_b]

        record = {
            "comparison_id": annotation["comparison_id"],
            "scenario_id": annotation["scenario_id"],
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
            "features_a": [
                features_a[name]
                for name in FEATURE_NAMES
            ],
            "features_b": [
                features_b[name]
                for name in FEATURE_NAMES
            ],
            # IMPORTANT:
            # Preserve the original human annotation exactly.
            "preferred_candidate": annotation["preferred_candidate"],
        }

        updated_records.append(record)

    # ------------------------------------------------------------
    # Save new dataset
    # ------------------------------------------------------------
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            updated_records,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    original_preferences = [
        item["preferred_candidate"]
        for item in annotations
    ]

    updated_preferences = [
        item["preferred_candidate"]
        for item in updated_records
    ]

    if original_preferences != updated_preferences:
        raise RuntimeError(
            "Human preference labels changed during regeneration."
        )

    print("Human preference labels: PRESERVED")

    for record in updated_records:
        assert len(record["features_a"]) == len(FEATURE_NAMES)
        assert len(record["features_b"]) == len(FEATURE_NAMES)

    print("Feature vector dimensions: VALID")
    print(f"Comparisons: {len(updated_records)}")

    print("\nOutput:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("PREFERENCE FEATURE REGENERATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()