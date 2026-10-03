import numpy as np


def calculate_path_length(x, y):
    """
    Calculate total trajectory length.

    Parameters
    ----------
    x, y : array-like
        Trajectory coordinates.

    Returns
    -------
    float
        Total path length.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return 0.0

    dx = np.diff(x)
    dy = np.diff(y)

    return float(
        np.sum(np.sqrt(dx ** 2 + dy ** 2))
    )


def calculate_trajectory_times(x, y, scenario):
    """
    Calculate physical time associated with each trajectory point.

    Time is calculated from cumulative trajectory distance
    and the scenario reference speed:

        t_i = cumulative_distance_i / reference_speed

    This keeps trajectory feature extraction consistent
    with the physical-time model used by MOPSO.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0:
        return np.array([], dtype=float)

    if len(x) == 1:
        return np.array([0.0], dtype=float)

    dx = np.diff(x)
    dy = np.diff(y)

    segment_lengths = np.sqrt(
        dx ** 2 + dy ** 2
    )

    cumulative_distance = np.concatenate(
        (
            [0.0],
            np.cumsum(segment_lengths),
        )
    )

    reference_speed = float(
        scenario.get("simulation", {}).get(
            "reference_speed",
            10.0,
        )
    )

    if reference_speed <= 0.0:
        raise ValueError(
            "Scenario reference_speed must be greater than zero."
        )

    return cumulative_distance / reference_speed


def calculate_obstacle_distances(x, y, scenario):
    """
    Calculate distances between the trajectory
    and moving vehicles.

    The obstacle position is evaluated using physical
    trajectory time rather than the trajectory sample index.

    Returns
    -------
    min_distance : float
        Minimum distance between the trajectory and obstacles.

    mean_distance : float
        Mean distance across all trajectory-obstacle evaluations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0:
        return np.inf, np.inf

    obstacles = scenario.get("obstacles", [])

    if not obstacles:
        return np.inf, np.inf

    times = calculate_trajectory_times(
        x,
        y,
        scenario,
    )

    all_distances = []

    for vehicle in obstacles:
        initial_position = np.asarray(
            vehicle["initial_position"],
            dtype=float,
        )

        velocity = np.asarray(
            vehicle.get(
                "velocity",
                [0.0, 0.0],
            ),
            dtype=float,
        )

        for px, py, time in zip(
            x,
            y,
            times,
        ):
            obstacle_position = (
                initial_position
                + velocity * time
            )

            distance = np.sqrt(
                (px - obstacle_position[0]) ** 2
                + (py - obstacle_position[1]) ** 2
            )

            all_distances.append(distance)

    if not all_distances:
        return np.inf, np.inf

    all_distances = np.asarray(
        all_distances,
        dtype=float,
    )

    return (
        float(np.min(all_distances)),
        float(np.mean(all_distances)),
    )


def calculate_smoothness(x, y):
    """
    Calculate a simple trajectory smoothness
    measure based on second spatial differences.

    Lower values indicate smoother trajectories.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return 0.0

    dx = np.diff(x)
    dy = np.diff(y)

    ddx = np.diff(dx)
    ddy = np.diff(dy)

    curvature_change = np.sqrt(
        ddx ** 2 + ddy ** 2
    )

    return float(
        np.mean(curvature_change)
    )


def calculate_max_curvature(x, y):
    """
    Approximate maximum curvature using
    discrete trajectory derivatives.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return 0.0

    dx = np.gradient(x)
    dy = np.gradient(y)

    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denominator = (
        dx ** 2 + dy ** 2
    ) ** 1.5

    numerator = np.abs(
        dx * ddy - dy * ddx
    )

    curvature = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-8,
    )

    return float(
        np.max(curvature)
    )


def calculate_clearance_features(
    min_obstacle_distance,
    scenario,
):
    """
    Calculate physical collision clearance and
    desired safety clearance.

    collision_clearance:
        Distance remaining after accounting for
        the modeled obstacle radius.

    desired_clearance:
        Distance remaining relative to the desired
        vehicle-to-vehicle safety distance.
    """
    obstacles = scenario.get("obstacles", [])

    if not obstacles:
        return {
            "collision_clearance": np.inf,
            "desired_clearance": np.inf,
        }

    obstacle_radius = max(
        float(
            obstacle.get(
                "radius",
                0.0,
            )
        )
        for obstacle in obstacles
    )

    desired_safety_distance = float(
        scenario.get("vehicle", {}).get(
            "safety_distance",
            0.0,
        )
    )

    collision_clearance = (
        min_obstacle_distance
        - obstacle_radius
    )

    desired_clearance = (
        min_obstacle_distance
        - desired_safety_distance
    )

    return {
        "collision_clearance": float(
            collision_clearance
        ),
        "desired_clearance": float(
            desired_clearance
        ),
    }


def extract_trajectory_features(x, y, scenario):
    """
    Extract the complete trajectory feature vector.
    """
    path_length = calculate_path_length(
        x,
        y,
    )

    min_distance, mean_distance = (
        calculate_obstacle_distances(
            x,
            y,
            scenario,
        )
    )

    clearance_features = (
        calculate_clearance_features(
            min_distance,
            scenario,
        )
    )

    smoothness = calculate_smoothness(
        x,
        y,
    )

    max_curvature = calculate_max_curvature(
        x,
        y,
    )

    return {
        "path_length": path_length,
        "min_obstacle_distance": min_distance,
        "mean_obstacle_distance": mean_distance,
        "collision_clearance": (
            clearance_features[
                "collision_clearance"
            ]
        ),
        "desired_clearance": (
            clearance_features[
                "desired_clearance"
            ]
        ),
        "smoothness": smoothness,
        "max_curvature": max_curvature,
    }