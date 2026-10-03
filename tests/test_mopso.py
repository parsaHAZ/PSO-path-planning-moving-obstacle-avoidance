import numpy as np

from data.scenarios.scenario_0001 import create_scenario
from research.mopso import MOPSOPathPlanner


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def create_planner(seed=42, n_particles=50, max_iterations=20, archive_size=20):
    scenario = create_scenario(seed=seed)

    planner = MOPSOPathPlanner(
        start=scenario["start"],
        end=scenario["end"],
        bounds_x=scenario["bounds_x"],
        bounds_y=scenario["bounds_y"],
        scenario=scenario,
        n_waypoints=4,
        n_particles=n_particles,
        max_iterations=max_iterations,
        inertia_weight=0.5,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        archive_size=archive_size,
        seed=seed,
    )

    return planner


def test_trajectory_time_model():
    print("\n" + "-" * 60)
    print("TEST: TRAJECTORY TIME MODEL")
    print("-" * 60)

    planner = create_planner()

    x = np.array([10.0, 20.0, 30.0, 40.0], dtype=float)
    y = np.array([10.0, 10.0, 10.0, 10.0], dtype=float)

    times = planner._trajectory_times(x, y)

    expected_distance = np.array([0.0, 10.0, 20.0, 30.0], dtype=float)
    reference_speed = planner.scenario["simulation"]["reference_speed"]
    expected_times = expected_distance / reference_speed

    print(f"Times: {times}")
    print(f"Expected: {expected_times}")

    check(np.allclose(times, expected_times), "Trajectory time calculation is incorrect.")
    check(np.isclose(times[0], 0.0), "Trajectory must start at t=0.")
    check(np.all(np.diff(times) >= 0.0), "Trajectory times must be monotonically increasing.")

    print("PASSED")


def test_moving_obstacle_position():
    print("\n" + "-" * 60)
    print("TEST: MOVING OBSTACLE POSITION")
    print("-" * 60)

    planner = create_planner()
    obstacle = planner.scenario["obstacles"][0]

    initial_position = np.asarray(obstacle["initial_position"], dtype=float)
    velocity = np.asarray(obstacle["velocity"], dtype=float)

    position_t0 = planner._moving_obstacle_position(obstacle, 0.0)
    position_t2 = planner._moving_obstacle_position(obstacle, 2.0)

    expected_t0 = initial_position
    expected_t2 = initial_position + velocity * 2.0

    print(f"t=0: {position_t0}")
    print(f"Expected: {expected_t0}")
    print(f"t=2: {position_t2}")
    print(f"Expected: {expected_t2}")

    check(np.allclose(position_t0, expected_t0), "Obstacle position at t=0 is incorrect.")
    check(np.allclose(position_t2, expected_t2), "Obstacle position at t=2 is incorrect.")

    print("PASSED")


def test_path_decoding():
    print("\n" + "-" * 60)
    print("TEST: PATH DECODING")
    print("-" * 60)

    planner = create_planner()

    particle = np.array(
        [25.0, 25.0, 40.0, 40.0, 60.0, 60.0, 75.0, 75.0],
        dtype=float,
    )

    x, y = planner._decode_particle(particle)

    print(f"Path points: {len(x)}")

    check(len(x) == 100, "Decoded path must contain 100 x points.")
    check(len(y) == 100, "Decoded path must contain 100 y points.")
    check(np.isclose(x[0], planner.start[0]), "Path does not start at the correct x coordinate.")
    check(np.isclose(y[0], planner.start[1]), "Path does not start at the correct y coordinate.")
    check(np.isclose(x[-1], planner.end[0]), "Path does not end at the correct x coordinate.")
    check(np.isclose(y[-1], planner.end[1]), "Path does not end at the correct y coordinate.")

    print("PASSED")


def test_objectives():
    print("\n" + "-" * 60)
    print("TEST: OBJECTIVE FUNCTIONS")
    print("-" * 60)

    planner = create_planner()

    particle = np.array(
        [25.0, 25.0, 40.0, 40.0, 60.0, 60.0, 75.0, 75.0],
        dtype=float,
    )

    path, objectives = planner.evaluate_particle(particle)
    x, y = path

    safety = objectives[0]
    efficiency = objectives[1]
    comfort = objectives[2]

    print(f"Safety: {safety:.8f}")
    print(f"Efficiency: {efficiency:.8f}")
    print(f"Comfort: {comfort:.8f}")

    check(len(x) == 100, "Objective path must contain 100 x points.")
    check(len(y) == 100, "Objective path must contain 100 y points.")
    check(objectives.shape == (3,), "Objective vector must contain 3 values.")
    check(np.all(np.isfinite(objectives)), "Objective values must be finite.")
    check(safety >= 0.0, "Safety objective cannot be negative.")
    check(efficiency >= 1.0 - 1e-10, "Efficiency objective should be >= 1.")
    check(comfort >= 0.0, "Comfort objective cannot be negative.")

    print("PASSED")


def test_pareto_dominance():
    print("\n" + "-" * 60)
    print("TEST: PARETO DOMINANCE")
    print("-" * 60)

    planner = create_planner()

    better = np.array([1.0, 2.0, 3.0])
    worse = np.array([2.0, 3.0, 4.0])
    mixed = np.array([0.5, 4.0, 2.0])

    check(planner.dominates(better, worse), "Better vector should dominate worse vector.")
    check(not planner.dominates(worse, better), "Worse vector must not dominate better vector.")
    check(not planner.dominates(better, mixed), "Mixed vector should not be dominated by better in all objectives.")
    check(not planner.dominates(mixed, better), "Mixed vector should not dominate better in all objectives.")

    print("PASSED")


def test_mopso_archive():
    print("\n" + "=" * 80)
    print("TEST: FULL MOPSO RUN")
    print("=" * 80)

    planner = create_planner(seed=42, n_particles=50, max_iterations=20, archive_size=20)
    results = planner.run()

    print(f"\nNumber of Pareto solutions: {len(results)}")
    print("\nCandidate               Safety        Efficiency        Comfort")
    print("-" * 80)

    for candidate in results:
        objectives = candidate["objectives"]
        print(
            f"{candidate['candidate_id']:<24}"
            f"{objectives[0]:>10.6f} "
            f"{objectives[1]:>16.6f} "
            f"{objectives[2]:>15.6f}"
        )

    check(len(results) > 0, "MOPSO returned an empty archive.")
    check(len(results) <= planner.archive_size, "Archive exceeds configured archive size.")

    for candidate in results:
        check("candidate_id" in candidate, "Candidate missing candidate_id.")
        check("path" in candidate, "Candidate missing path.")
        check("objectives" in candidate, "Candidate missing objectives.")
        check("position" in candidate, "Candidate missing position.")
        check("seed" in candidate, "Candidate missing seed.")

        objectives = candidate["objectives"]

        check(isinstance(objectives, np.ndarray), "Objectives must be a NumPy array.")
        check(objectives.shape == (3,), "Objective vector must contain exactly 3 values.")
        check(np.all(np.isfinite(objectives)), "Candidate contains non-finite objective values.")
        check(objectives[0] >= 0.0, "Safety objective cannot be negative.")
        check(objectives[1] >= 1.0 - 1e-10, "Efficiency objective should be >= 1.")
        check(objectives[2] >= 0.0, "Comfort objective cannot be negative.")

        x, y = candidate["path"]

        check(len(x) == 100, "Candidate path must contain 100 x points.")
        check(len(y) == 100, "Candidate path must contain 100 y points.")

    for i in range(len(results)):
        for j in range(len(results)):
            if i == j:
                continue

            check(
                not planner.dominates(results[j]["objectives"], results[i]["objectives"]),
                f"Archive contains a dominated solution: {results[i]['candidate_id']}",
            )

    print("\nAll archived solutions are mutually non-dominated.")
    print("\nFULL MOPSO TEST PASSED")
    print("=" * 80)


def test_deterministic_results():
    print("\n" + "-" * 60)
    print("TEST: MOPSO DETERMINISM")
    print("-" * 60)

    planner_a = create_planner(seed=42, n_particles=30, max_iterations=10, archive_size=20)
    planner_b = create_planner(seed=42, n_particles=30, max_iterations=10, archive_size=20)

    results_a = planner_a.run()
    results_b = planner_b.run()

    check(len(results_a) == len(results_b), "Repeated runs produced different archive sizes.")

    objectives_a = np.array([candidate["objectives"] for candidate in results_a])
    objectives_b = np.array([candidate["objectives"] for candidate in results_b])

    check(np.allclose(objectives_a, objectives_b), "MOPSO is not deterministic for the same seed.")

    print(f"Archive size: {len(results_a)}")
    print("Repeated runs produced identical objective values.")
    print("PASSED")


def main():
    print("\n" + "=" * 80)
    print("MOPSO STANDALONE TEST SUITE")
    print("=" * 80)

    tests = [
        ("Trajectory time model", test_trajectory_time_model),
        ("Moving obstacle position", test_moving_obstacle_position),
        ("Path decoding", test_path_decoding),
        ("Objective functions", test_objectives),
        ("Pareto dominance", test_pareto_dominance),
        ("MOPSO archive", test_mopso_archive),
        ("MOPSO determinism", test_deterministic_results),
    ]

    passed = 0

    for name, test_function in tests:
        try:
            test_function()
            passed += 1
        except Exception as error:
            print(f"\nFAILED: {name}")
            print(f"Reason: {error}")
            raise

    print("\n" + "=" * 80)
    print(f"ALL MOPSO TESTS PASSED ({passed}/{len(tests)})")
    print("=" * 80)


if __name__ == "__main__":
    main()