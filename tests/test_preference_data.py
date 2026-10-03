import numpy as np

from research.preference_data import (
    FEATURE_NAMES,
    feature_vector,
    create_preference_pair,
    preference_to_label,
)


def test_feature_vector():
    features = {
        "path_length": 113.074,
        "min_obstacle_distance": 0.6746,
        "mean_obstacle_distance": 28.838,
        "collision_clearance": -4.3254,
        "desired_clearance": -7.3254,
        "smoothness": 0.00146,
        "max_curvature": 0.00088,
    }

    vector = feature_vector(features)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (7,)
    assert vector.dtype == float

    expected = np.array(
        [
            113.074,
            0.6746,
            28.838,
            -4.3254,
            -7.3254,
            0.00146,
            0.00088,
        ],
        dtype=float,
    )

    assert np.allclose(vector, expected)

    print("Feature vector:")
    print(vector)
    print("PASSED")


def test_missing_feature():
    features = {
        "path_length": 113.074,
        "min_obstacle_distance": 0.6746,
        "mean_obstacle_distance": 28.838,
        "collision_clearance": -4.3254,
        "desired_clearance": -7.3254,
        "smoothness": 0.00146,
        # max_curvature intentionally missing
    }

    try:
        feature_vector(features)
    except KeyError:
        print("Missing-feature validation: PASSED")
        return

    raise AssertionError(
        "feature_vector() should raise KeyError when a feature is missing."
    )


def test_preference_pair():
    features_a = {
        "path_length": 113.074,
        "min_obstacle_distance": 0.6746,
        "mean_obstacle_distance": 28.838,
        "collision_clearance": -4.3254,
        "desired_clearance": -7.3254,
        "smoothness": 0.00146,
        "max_curvature": 0.00088,
    }

    features_b = {
        "path_length": 115.226,
        "min_obstacle_distance": 9.1749,
        "mean_obstacle_distance": 30.376,
        "collision_clearance": 4.1749,
        "desired_clearance": 1.1749,
        "smoothness": 0.00880,
        "max_curvature": 0.01364,
    }

    pair = create_preference_pair(
        candidate_a="candidate_01",
        features_a=features_a,
        candidate_b="candidate_04",
        features_b=features_b,
        preferred_candidate="candidate_04",
    )

    assert pair["candidate_a"] == "candidate_01"
    assert pair["candidate_b"] == "candidate_04"
    assert pair["preferred_candidate"] == "candidate_04"

    assert pair["features_a"].shape == (7,)
    assert pair["features_b"].shape == (7,)

    print("Preference pair:")
    print(f"  A: {pair['candidate_a']}")
    print(f"  B: {pair['candidate_b']}")
    print(f"  Preferred: {pair['preferred_candidate']}")
    print("PASSED")


def test_preference_labels():
    assert (
        preference_to_label(
            "candidate_01",
            "candidate_01",
            "candidate_04",
        )
        == 1.0
    )

    assert (
        preference_to_label(
            "candidate_04",
            "candidate_01",
            "candidate_04",
        )
        == 0.0
    )

    assert (
        preference_to_label(
            "tie",
            "candidate_01",
            "candidate_04",
        )
        == 0.5
    )

    print("Preference label conversion: PASSED")


def test_invalid_preference():
    try:
        create_preference_pair(
            candidate_a="candidate_01",
            features_a={name: 0.0 for name in FEATURE_NAMES},
            candidate_b="candidate_02",
            features_b={name: 0.0 for name in FEATURE_NAMES},
            preferred_candidate="candidate_99",
        )
    except ValueError:
        print("Invalid preference validation: PASSED")
        return

    raise AssertionError(
        "create_preference_pair() should reject an invalid candidate ID."
    )


def main():
    print("=" * 70)
    print("PREFERENCE DATA TEST SUITE")
    print("=" * 70)

    print("\n------------------------------------------------------------")
    print("TEST: FEATURE VECTOR")
    print("------------------------------------------------------------")
    test_feature_vector()

    print("\n------------------------------------------------------------")
    print("TEST: MISSING FEATURE VALIDATION")
    print("------------------------------------------------------------")
    test_missing_feature()

    print("\n------------------------------------------------------------")
    print("TEST: PREFERENCE PAIR")
    print("------------------------------------------------------------")
    test_preference_pair()

    print("\n------------------------------------------------------------")
    print("TEST: PREFERENCE LABELS")
    print("------------------------------------------------------------")
    test_preference_labels()

    print("\n------------------------------------------------------------")
    print("TEST: INVALID PREFERENCE")
    print("------------------------------------------------------------")
    test_invalid_preference()

    print("\n" + "=" * 70)
    print("ALL PREFERENCE DATA TESTS PASSED (5/5)")
    print("=" * 70)


if __name__ == "__main__":
    main()