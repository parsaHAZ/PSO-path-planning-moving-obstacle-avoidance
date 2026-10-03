from pathlib import Path

from pso import PSOPathPlanner

from data.scenarios.scenario_0001 import create_scenario
from research.preference_dataset import (
    generate_preference_pairs,
    save_preference_dataset_csv,
    save_preference_dataset_json,
)


def main():
    print("=" * 80)
    print("PREFERENCE DATASET GENERATION TEST")
    print("=" * 80)

    scenario = create_scenario()

    planner = PSOPathPlanner(
        start=scenario["start"],
        end=scenario["end"],
        n_obstacles=0,
        max_iterations=50,
        bounds_x=scenario["bounds_x"],
        bounds_y=scenario["bounds_y"],
        scenario=scenario,
    )

    candidates = planner.generate_candidates(n_candidates=5)

    print()
    print(f"Candidates generated: {len(candidates)}")

    pairs = generate_preference_pairs(candidates, scenario)
    print(f"Pairwise comparisons generated: {len(pairs)}")

    expected_pairs = len(candidates) * (len(candidates) - 1) // 2
    print(f"Expected comparisons: {expected_pairs}")

    print()
    for pair in pairs:
        print(
            f"{pair['comparison_id']} | "
            f"{pair['candidate_a']} vs "
            f"{pair['candidate_b']} | "
            f"preference={pair['preferred_candidate']}"
        )

    output_directory = Path("data/preference")
    json_path = output_directory / "scenario_0001_preferences.json"
    csv_path = output_directory / "scenario_0001_preferences.csv"

    save_preference_dataset_json(pairs, json_path)
    save_preference_dataset_csv(pairs, csv_path)

    print()
    print(f"JSON dataset saved to: {json_path}")
    print(f"CSV dataset saved to: {csv_path}")

    print()
    print("Preference dataset generated successfully.")


if __name__ == "__main__":
    main()