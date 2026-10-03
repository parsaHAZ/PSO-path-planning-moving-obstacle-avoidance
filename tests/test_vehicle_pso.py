import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.scenarios.scenario_0001 import create_scenario
from pso import PSOPathPlanner


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    print("\n" + "=" * 70)
    print("VEHICLE SCENARIO + PSO TEST")
    print("=" * 70)

    # Load scenario
    scenario = create_scenario(seed=42)

    # Create planner
    # PSO parameters such as n_waypoints, n_particles, w, c1,
    # and c2 are defined internally by PSOPathPlanner.
    planner = PSOPathPlanner(
        start=scenario["start"],
        end=scenario["end"],
        n_obstacles=len(scenario.get("obstacles", [])),
        max_iterations=50,
        bounds_x=scenario["bounds_x"],
        bounds_y=scenario["bounds_y"],
        scenario=scenario,
    )

    # Verify internal PSO parameters
    check(
        planner.n_waypoints == 4,
        f"Unexpected number of waypoints: {planner.n_waypoints}",
    )
    check(
        planner.n_particles == 100,
        f"Unexpected number of particles: {planner.n_particles}",
    )
    check(
        np.isclose(planner.w, 0.5),
        f"Unexpected inertia weight: {planner.w}",
    )
    check(
        np.isclose(planner.c1, 1.5),
        f"Unexpected cognitive coefficient: {planner.c1}",
    )
    check(
        np.isclose(planner.c2, 1.5),
        f"Unexpected social coefficient: {planner.c2}",
    )

    # Run PSO
    path, best_cost, best_iteration, convergence = planner.run()

    # Result structure
    check(path is not None, "PSO returned no valid path.")
    check(
        isinstance(path, (tuple, list)),
        "PSO path must be a tuple or list.",
    )
    check(
        len(path) == 2,
        "PSO path must contain x and y arrays.",
    )

    # Extract path
    x, y = path
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    best_cost = float(best_cost)
    best_iteration = int(best_iteration)

    # Print results
    print(f"Scenario: {scenario['scenario_id']}")
    print(f"Best cost: {best_cost:.4f}")
    print(f"Best iteration: {best_iteration}")
    print(f"Convergence points: {len(convergence)}")
    print(f"Path points: {len(x)}")
    print(f"Start point: ({x[0]:.2f}, {y[0]:.2f})")
    print(f"End point: ({x[-1]:.2f}, {y[-1]:.2f})")

    # Numerical checks
    check(
        np.isfinite(best_cost),
        "Best PSO cost is not finite.",
    )
    check(
        best_cost < 1e6,
        f"PSO did not find a valid path. Returned penalty cost: {best_cost}",
    )
    check(
        len(x) == 100,
        f"PSO path must contain 100 x points, got {len(x)}.",
    )
    check(
        len(y) == 100,
        f"PSO path must contain 100 y points, got {len(y)}.",
    )
    check(
        len(convergence) == 50,
        f"PSO convergence history must contain 50 points, got {len(convergence)}.",
    )
    check(
        0 <= best_iteration < 50,
        f"Best iteration is outside the expected range: {best_iteration}.",
    )

    # Path numerical validity
    check(
        np.all(np.isfinite(x)),
        "PSO x path contains non-finite values.",
    )
    check(
        np.all(np.isfinite(y)),
        "PSO y path contains non-finite values.",
    )

    # Check convergence values
    convergence_array = np.asarray(convergence, dtype=float)

    check(
        np.all(np.isfinite(convergence_array)),
        "PSO convergence contains non-finite values.",
    )

    check(
        np.isclose(convergence_array[-1], best_cost),
        (
            "Final convergence value does not match best cost.\n"
            f"Final convergence: {convergence_array[-1]}\n"
            f"Best cost: {best_cost}"
        ),
    )

    # Check path bounds
    bounds_x = scenario["bounds_x"]
    bounds_y = scenario["bounds_y"]

    check(
        np.all(x >= bounds_x[0]) and np.all(x <= bounds_x[1]),
        "PSO path contains x values outside the scenario bounds.",
    )

    check(
        np.all(y >= bounds_y[0]) and np.all(y <= bounds_y[1]),
        "PSO path contains y values outside the scenario bounds.",
    )

    # Check start point
    # The current PSO uses a smoothed B-spline:
    #     splprep(..., s=2.0)
    #
    # Therefore the evaluated spline does not have to pass exactly
    # through the first control point. We validate the Euclidean
    # distance from the expected start point instead of requiring
    # exact equality.

    actual_start = np.asarray(
        [x[0], y[0]],
        dtype=float,
    )

    expected_start = np.asarray(
        scenario["start"],
        dtype=float,
    )

    start_error = np.linalg.norm(
        actual_start - expected_start
    )

    print(f"Start point error: {start_error:.6f}")

    check(
        start_error <= 0.25,
        (
            "PSO path start point deviates too far "
            "from the scenario start point.\n"
            f"Expected: {expected_start}\n"
            f"Actual:   {actual_start}\n"
            f"Distance error: {start_error:.6f}"
        ),
    )

    # Check goal point
    # The same B-spline smoothing applies to the final control point.
    # Therefore the evaluated spline may deviate slightly from the
    # exact scenario goal point.

    actual_end = np.asarray(
        [x[-1], y[-1]],
        dtype=float,
    )

    expected_end = np.asarray(
        scenario["end"],
        dtype=float,
    )

    end_error = np.linalg.norm(
        actual_end - expected_end
    )

    print(f"End point error: {end_error:.6f}")

    check(
        end_error <= 0.25,
        (
            "PSO path end point deviates too far "
            "from the scenario goal point.\n"
            f"Expected: {expected_end}\n"
            f"Actual:   {actual_end}\n"
            f"Distance error: {end_error:.6f}"
        ),
    )

    # Success
    print("\n" + "-" * 70)
    print("ALL PSO VEHICLE CHECKS PASSED")
    print("-" * 70)
    print(f"Best cost: {best_cost:.4f}")
    print(f"Best iteration: {best_iteration}")
    print(f"Path points: {len(x)}")
    print(f"Convergence points: {len(convergence)}")
    print(f"Start error: {start_error:.6f}")
    print(f"End error: {end_error:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()