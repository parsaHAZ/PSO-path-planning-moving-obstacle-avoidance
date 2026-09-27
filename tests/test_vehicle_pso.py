import numpy as np

from pso import PSOPathPlanner
from data.scenarios.scenario_0001 import create_scenario


def main():
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

    result = planner.run()
    path, cost, best_iteration, convergence = result

    print()
    print("=" * 50)
    print("VEHICLE SCENARIO + PSO TEST")
    print("=" * 50)
    print(f"Scenario: {scenario['scenario_id']}")
    print(f"Best cost: {cost:.4f}")
    print(f"Best iteration: {best_iteration}")
    print(f"Convergence points: {len(convergence)}")

    if path is None:
        print("No valid path found.")
    else:
        x, y = path
        print(f"Path points: {len(x)}")
        print(f"Start point: ({x[0]:.2f}, {y[0]:.2f})")
        print(f"End point: ({x[-1]:.2f}, {y[-1]:.2f})")
        print()
        print("Vehicle scenario successfully connected to PSO.")


if __name__ == "__main__":
    main()