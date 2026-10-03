"""
Preference data utilities for trajectory preference learning.

This module converts trajectory feature dictionaries into fixed-order
numerical vectors and creates pairwise preference samples.

Preference labels can represent:
    - candidate_a
    - candidate_b
    - tie
"""

import numpy as np


FEATURE_NAMES = [
    "path_length",
    "min_obstacle_distance",
    "mean_obstacle_distance",
    "collision_clearance",
    "desired_clearance",
    "smoothness",
    "max_curvature",
]


def feature_vector(features):
    """
    Convert a trajectory feature dictionary into a fixed-order
    numerical feature vector.

    Parameters
    ----------
    features : dict
        Dictionary containing all FEATURE_NAMES.

    Returns
    -------
    np.ndarray
        Feature vector in the fixed FEATURE_NAMES order.
    """

    missing = [name for name in FEATURE_NAMES if name not in features]

    if missing:
        raise KeyError(
            f"Missing required trajectory features: {missing}"
        )

    return np.array(
        [features[name] for name in FEATURE_NAMES],
        dtype=float,
    )


def create_preference_pair(
    candidate_a,
    features_a,
    candidate_b,
    features_b,
    preferred_candidate=None,
):
    """
    Create one pairwise preference sample.

    Parameters
    ----------
    candidate_a : str
        ID of the first trajectory candidate.

    features_a : dict
        Feature dictionary for candidate A.

    candidate_b : str
        ID of the second trajectory candidate.

    features_b : dict
        Feature dictionary for candidate B.

    preferred_candidate : str or None
        One of:
            - candidate_a
            - candidate_b
            - "tie"
            - None

        None means the preference has not been assigned yet.

    Returns
    -------
    dict
        Pairwise preference sample.
    """

    valid_preferences = {
        None,
        candidate_a,
        candidate_b,
        "tie",
    }

    if preferred_candidate not in valid_preferences:
        raise ValueError(
            "preferred_candidate must be candidate_a, "
            "candidate_b, 'tie', or None."
        )

    return {
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "features_a": feature_vector(features_a),
        "features_b": feature_vector(features_b),
        "preferred_candidate": preferred_candidate,
    }


def preference_to_label(
    preference,
    candidate_a,
    candidate_b,
):
    """
    Convert a candidate preference into a numerical label.

    Returns
    -------
    float
        1.0 -> candidate_a preferred
        0.0 -> candidate_b preferred
        0.5 -> tie
    """

    if preference == candidate_a:
        return 1.0

    if preference == candidate_b:
        return 0.0

    if preference == "tie":
        return 0.5

    raise ValueError(
        "Preference must match candidate_a, "
        "candidate_b, or be 'tie'."
    )