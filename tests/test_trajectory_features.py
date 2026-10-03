import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scenarios.scenario_0001 import create_scenario
from pso import PSOPathPlanner
from research.trajectory_features import (
    calculate_path_length,
    calculate_trajectory_times,
    calculate_obstacle_distances,
    calculate_smoothness,
    calculate_max_curvature,
    calculate_clearance_features,
    extract_trajectory_features,
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_basic_geometry():
    print()
    print("-" * 70)
    print("TEST: BASIC PATH LENGTH")
    print("-" * 70)

    x = np.array([0.0, 3.0, 6.0])
    y = np.array([0.0, 4.0, 8.0])

    # Two segments, each with length 5.
    expected = 10.0

    result = calculate_path_length(x, y)

    print(f"Calculated path length: {result:.6f}")
    print(f"Expected path length:   {expected:.6f}")

    check(
        np.isclose(result, expected),
        (
            "Path length calculation is incorrect.\n"
            f"Expected: {expected}\n"
            f"Actual:   {result}"
        ),
    )

    print("PASSED")


def test_trajectory_time_model():
    print()
    print("-" * 70)
    print("TEST: TRAJECTORY TIME MODEL")
    print("-" * 70)

    scenario = create_scenario()

    # Reference speed = 10 units/s.
    #
    # Positions:
    # (0,0) -> (10,0) -> (20,0) -> (30,0)
    #
    # Cumulative distances:
    # 0, 10, 20, 30
    #
    # Expected times:
    # 0, 1, 2, 3 seconds.

    x = np.array([0.0, 10.0, 20.0, 30.0])
    y = np.array([0.0, 0.0, 0.0, 0.0])

    times = calculate_trajectory_times(
        x,
        y,
        scenario,
    )

    expected_times = np.array(
        [0.0, 1.0, 2.0, 3.0]
    )

    print(f"Calculated times: {times}")
    print(f"Expected times:   {expected_times}")

    check(
        np.allclose(times, expected_times),
        (
            "Trajectory time model is incorrect.\n"
            f"Expected: {expected_times}\n"
            f"Actual:   {times}"
        ),
    )

    print("PASSED")


def test_moving_obstacle_distance():
    print()
    print("-" * 70)
    print("TEST: MOVING OBSTACLE DISTANCE MODEL")
    print("-" * 70)

    scenario = create_scenario()

    # Construct a trajectory along x = 45.
    #
    # Obstacle:
    # initial position = [45, 45]
    # velocity = [0, 0.15]
    #
    # At t = 0:
    # obstacle = [45, 45]
    #
    # At t = 1:
    # obstacle = [45, 45.15]
    #
    # At t = 2:
    # obstacle = [45, 45.30]

    x = np.array(
        [45.0, 45.0, 45.0]
    )

    y = np.array(
        [45.0, 55.0, 65.0]
    )

    min_distance, mean_distance = (
        calculate_obstacle_distances(
            x,
            y,
            scenario,
        )
    )

    expected_distances = np.array(
        [
            0.0,
            9.85,
            19.70,
        ]
    )

    expected_minimum = np.min(
        expected_distances
    )

    expected_mean = np.mean(
        expected_distances
    )

    print(f"Calculated minimum distance: {min_distance:.6f}")
    print(f"Expected minimum distance:   {expected_minimum:.6f}")
    print(f"Calculated mean distance:    {mean_distance:.6f}")
    print(f"Expected mean distance:      {expected_mean:.6f}")

    check(
        np.isclose(
            min_distance,
            expected_minimum,
        ),
        (
            "Minimum obstacle distance is incorrect.\n"
            f"Expected: {expected_minimum}\n"
            f"Actual:   {min_distance}"
        ),
    )

    check(
        np.isclose(
            mean_distance,
            expected_mean,
        ),
        (
            "Mean obstacle distance is incorrect.\n"
            f"Expected: {expected_mean}\n"
            f"Actual:   {mean_distance}"
        ),
    )

    print("PASSED")


def test_smoothness_and_curvature():
    print()
    print("-" * 70)
    print("TEST: SMOOTHNESS AND CURVATURE")
    print("-" * 70)

    # Perfectly straight trajectory.
    x = np.linspace(
        0.0,
        10.0,
        100,
    )

    y = np.zeros(100)

    smoothness = calculate_smoothness(
        x,
        y,
    )

    max_curvature = calculate_max_curvature(
        x,
        y,
    )

    print(f"Straight-path smoothness: {smoothness:.12f}")
    print(f"Straight-path curvature:  {max_curvature:.12f}")

    check(
        np.isclose(smoothness, 0.0, atol=1e-10),
        (
            "Straight trajectory should have approximately "
            "zero smoothness penalty.\n"
            f"Actual: {smoothness}"
        ),
    )

    check(
        np.isclose(max_curvature, 0.0, atol=1e-10),
        (
            "Straight trajectory should have approximately "
            "zero curvature.\n"
            f"Actual: {max_curvature}"
        ),
    )

    print("PASSED")


def test_clearance_features():
    print()
    print("-" * 70)
    print("TEST: CLEARANCE FEATURES")
    print("-" * 70)

    scenario = create_scenario()

    # Scenario:
    # obstacle radius = 5
    # desired safety distance = 8
    #
    # If minimum distance = 20:
    # collision clearance = 20 - 5 = 15
    # desired clearance   = 20 - 8 = 12

    min_distance = 20.0

    features = calculate_clearance_features(
        min_distance,
        scenario,
    )

    expected_collision_clearance = 15.0
    expected_desired_clearance = 12.0

    print(
        "Collision clearance:",
        f"{features['collision_clearance']:.6f}",
    )

    print(
        "Expected collision clearance:",
        f"{expected_collision_clearance:.6f}",
    )

    print(
        "Desired clearance:",
        f"{features['desired_clearance']:.6f}",
    )

    print(
        "Expected desired clearance:",
        f"{expected_desired_clearance:.6f}",
    )

    check(
        np.isclose(
            features["collision_clearance"],
            expected_collision_clearance,
        ),
        "Collision clearance calculation is incorrect.",
    )

    check(
        np.isclose(
            features["desired_clearance"],
            expected_desired_clearance,
        ),
        "Desired clearance calculation is incorrect.",
    )

    print("PASSED")


def test_complete_feature_extraction():
    print()
    print("-" * 70)
    print("TEST: COMPLETE FEATURE EXTRACTION")
    print("-" * 70)

    scenario = create_scenario()

    planner = PSOPathPlanner(
        start=scenario["start"],
        end=scenario["end"],
        n_obstacles=len(
            scenario.get("obstacles", [])
        ),
        max_iterations=50,
        bounds_x=scenario["bounds_x"],
        bounds_y=scenario["bounds_y"],
        scenario=scenario,
    )

    candidates = planner.generate_candidates(
        n_candidates=5
    )

    check(
        len(candidates) == 5,
        (
            "Expected 5 PSO candidates for feature extraction.\n"
            f"Actual: {len(candidates)}"
        ),
    )

    required_features = {
        "path_length",
        "min_obstacle_distance",
        "mean_obstacle_distance",
        "collision_clearance",
        "desired_clearance",
        "smoothness",
        "max_curvature",
    }

    for candidate in candidates:
        x, y = candidate["path"]

        features = extract_trajectory_features(
            x,
            y,
            scenario,
        )

        print()
        print(candidate["candidate_id"])

        for feature_name in sorted(features.keys()):
            print(
                f"  {feature_name}: "
                f"{features[feature_name]:.8f}"
            )

        # Check feature names.
        check(
            set(features.keys()) == required_features,
            (
                f"{candidate['candidate_id']}: "
                "feature dictionary does not contain "
                "the expected feature set."
            ),
        )

        # Check numerical validity.
        for feature_name, value in features.items():
            check(
                np.isfinite(value),
                (
                    f"{candidate['candidate_id']}: "
                    f"feature '{feature_name}' is not finite."
                ),
            )

        # Basic physical sanity checks.
        check(
            features["path_length"] > 0.0,
            f"{candidate['candidate_id']}: path length must be positive.",
        )

        check(
            features["min_obstacle_distance"] >= 0.0,
            (
                f"{candidate['candidate_id']}: "
                "minimum obstacle distance cannot be negative."
            ),
        )

        check(
            features["mean_obstacle_distance"] >= 0.0,
            (
                f"{candidate['candidate_id']}: "
                "mean obstacle distance cannot be negative."
            ),
        )

        check(
            features["smoothness"] >= 0.0,
            (
                f"{candidate['candidate_id']}: "
                "smoothness cannot be negative."
            ),
        )

        check(
            features["max_curvature"] >= 0.0,
            (
                f"{candidate['candidate_id']}: "
                "maximum curvature cannot be negative."
            ),
        )

    print()
    print("All PSO candidates produced valid feature vectors.")
    print("PASSED")


def main():
    print()
    print("=" * 80)
    print("TRAJECTORY FEATURE EXTRACTION TEST SUITE")
    print("=" * 80)

    tests = [
        (
            "Basic geometry",
            test_basic_geometry,
        ),
        (
            "Trajectory time model",
            test_trajectory_time_model,
        ),
        (
            "Moving obstacle distance",
            test_moving_obstacle_distance,
        ),
        (
            "Smoothness and curvature",
            test_smoothness_and_curvature,
        ),
        (
            "Clearance features",
            test_clearance_features,
        ),
        (
            "Complete feature extraction",
            test_complete_feature_extraction,
        ),
    ]

    passed = 0

    for name, test_function in tests:
        test_function()
        passed += 1

    print()
    print("=" * 80)
    print(
        f"ALL TRAJECTORY FEATURE TESTS PASSED "
        f"({passed}/{len(tests)})"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()