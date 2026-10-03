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
    print("PSO CANDIDATE GENERATION TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load scenario
    # ------------------------------------------------------------
    scenario = create_scenario(seed=42)

    # ------------------------------------------------------------
    # Create planner
    # ------------------------------------------------------------
    planner = PSOPathPlanner(
        start=scenario["start"],
        end=scenario["end"],
        n_obstacles=len(scenario.get("obstacles", [])),
        max_iterations=50,
        bounds_x=scenario["bounds_x"],
        bounds_y=scenario["bounds_y"],
        scenario=scenario,
    )

    # ------------------------------------------------------------
    # Generate candidates
    # ------------------------------------------------------------
    requested_candidates = 5

    candidates = planner.generate_candidates(
        n_candidates=requested_candidates
    )

    print(f"Scenario: {scenario['scenario_id']}")
    print(f"Candidates requested: {requested_candidates}")
    print(f"Candidates generated: {len(candidates)}")
    print()

    # ------------------------------------------------------------
    # Basic result checks
    # ------------------------------------------------------------
    check(
        isinstance(candidates, list),
        "Candidate generation must return a list.",
    )

    check(
        len(candidates) == requested_candidates,
        (
            f"Expected {requested_candidates} candidates, "
            f"got {len(candidates)}."
        ),
    )

    # ------------------------------------------------------------
    # Expected deterministic seeds
    # ------------------------------------------------------------
    expected_seeds = [42, 43, 44, 45, 46]

    actual_seeds = [
        int(candidate["seed"])
        for candidate in candidates
    ]

    check(
        actual_seeds == expected_seeds,
        (
            "Candidate seeds do not match the expected deterministic "
            f"sequence.\nExpected: {expected_seeds}\n"
            f"Actual:   {actual_seeds}"
        ),
    )

    # ------------------------------------------------------------
    # Candidate-specific validation
    # ------------------------------------------------------------
    required_fields = {
        "candidate_id",
        "path",
        "cost",
        "best_iteration",
        "convergence",
        "seed",
    }

    candidate_ids = []
    candidate_seeds = []

    for candidate in candidates:

        candidate_id = candidate["candidate_id"]
        candidate_ids.append(candidate_id)
        candidate_seeds.append(candidate["seed"])

        print(
            f"{candidate_id} | "
            f"seed={candidate['seed']} | "
            f"cost={float(candidate['cost']):.4f} | "
            f"points={len(candidate['path'][0])} | "
            f"best_iteration={candidate['best_iteration']}"
        )

        # Required fields
        check(
            required_fields.issubset(candidate.keys()),
            (
                f"{candidate_id}: missing required fields.\n"
                f"Required: {sorted(required_fields)}\n"
                f"Available: {sorted(candidate.keys())}"
            ),
        )

        # --------------------------------------------------------
        # Path structure
        # --------------------------------------------------------
        path = candidate["path"]

        check(
            isinstance(path, (tuple, list)),
            f"{candidate_id}: path must be a tuple or list.",
        )

        check(
            len(path) == 2,
            f"{candidate_id}: path must contain x and y arrays.",
        )

        x, y = path

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        check(
            len(x) == 100,
            (
                f"{candidate_id}: path must contain 100 x points, "
                f"got {len(x)}."
            ),
        )

        check(
            len(y) == 100,
            (
                f"{candidate_id}: path must contain 100 y points, "
                f"got {len(y)}."
            ),
        )

        check(
            np.all(np.isfinite(x)),
            f"{candidate_id}: x path contains non-finite values.",
        )

        check(
            np.all(np.isfinite(y)),
            f"{candidate_id}: y path contains non-finite values.",
        )

        # --------------------------------------------------------
        # Cost
        # --------------------------------------------------------
        cost = float(candidate["cost"])

        check(
            np.isfinite(cost),
            f"{candidate_id}: cost is not finite.",
        )

        check(
            cost < 1e6,
            (
                f"{candidate_id}: invalid PSO candidate. "
                f"Penalty cost returned: {cost}"
            ),
        )

        # --------------------------------------------------------
        # Best iteration
        # --------------------------------------------------------
        best_iteration = int(candidate["best_iteration"])

        check(
            0 <= best_iteration < 50,
            (
                f"{candidate_id}: best iteration is outside "
                f"the expected range: {best_iteration}"
            ),
        )

        # --------------------------------------------------------
        # Convergence
        # --------------------------------------------------------
        convergence = np.asarray(
            candidate["convergence"],
            dtype=float,
        )

        check(
            len(convergence) == 50,
            (
                f"{candidate_id}: convergence history must contain "
                f"50 points, got {len(convergence)}."
            ),
        )

        check(
            np.all(np.isfinite(convergence)),
            f"{candidate_id}: convergence contains non-finite values.",
        )

        check(
            np.isclose(convergence[-1], cost),
            (
                f"{candidate_id}: final convergence value does not "
                f"match candidate cost.\n"
                f"Final convergence: {convergence[-1]}\n"
                f"Candidate cost:   {cost}"
            ),
        )

        # --------------------------------------------------------
        # Bounds
        # --------------------------------------------------------
        bounds_x = scenario["bounds_x"]
        bounds_y = scenario["bounds_y"]

        check(
            np.all(x >= bounds_x[0]) and np.all(x <= bounds_x[1]),
            f"{candidate_id}: x path contains values outside bounds.",
        )

        check(
            np.all(y >= bounds_y[0]) and np.all(y <= bounds_y[1]),
            f"{candidate_id}: y path contains values outside bounds.",
        )

        # --------------------------------------------------------
        # Start point
        #
        # The PSO uses:
        #
        #     splprep(..., s=2.0)
        #
        # Therefore the evaluated spline can deviate slightly
        # from the first control point.
        # --------------------------------------------------------
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

        check(
            start_error <= 0.25,
            (
                f"{candidate_id}: path does not start sufficiently "
                f"close to the scenario start point.\n"
                f"Expected: {expected_start}\n"
                f"Actual:   {actual_start}\n"
                f"Distance error: {start_error:.6f}"
            ),
        )

        # --------------------------------------------------------
        # Goal point
        #
        # The same spline smoothing applies to the final point.
        # --------------------------------------------------------
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

        check(
            end_error <= 0.25,
            (
                f"{candidate_id}: path does not end sufficiently "
                f"close to the scenario goal point.\n"
                f"Expected: {expected_end}\n"
                f"Actual:   {actual_end}\n"
                f"Distance error: {end_error:.6f}"
            ),
        )

    # ------------------------------------------------------------
    # Candidate ID uniqueness
    # ------------------------------------------------------------
    check(
        len(set(candidate_ids)) == len(candidate_ids),
        "Candidate IDs are not unique.",
    )

    # ------------------------------------------------------------
    # Candidate seed uniqueness
    # ------------------------------------------------------------
    check(
        len(set(candidate_seeds)) == len(candidate_seeds),
        "Candidate seeds are not unique.",
    )

    # ------------------------------------------------------------
    # Expected candidate IDs
    # ------------------------------------------------------------
    expected_ids = [
        f"candidate_{i:02d}"
        for i in range(1, requested_candidates + 1)
    ]

    check(
        candidate_ids == expected_ids,
        (
            "Candidate IDs do not match the expected sequence.\n"
            f"Expected: {expected_ids}\n"
            f"Actual:   {candidate_ids}"
        ),
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    costs = [
        float(candidate["cost"])
        for candidate in candidates
    ]

    print()
    print(f"Minimum cost: {min(costs):.4f}")
    print(f"Maximum cost: {max(costs):.4f}")
    print("Multiple PSO trajectory candidates generated successfully.")

    print("\n" + "=" * 70)
    print("PSO CANDIDATE GENERATION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()