import numpy as np

from data.scenarios.scenario_0001 import create_scenario


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    print("\n" + "=" * 60)
    print("VEHICLE SCENARIO TEST")
    print("=" * 60)

    scenario = create_scenario(seed=42)

    print(f"Scenario ID: {scenario['scenario_id']}")
    print(f"Start: {scenario['start']}")
    print(f"End: {scenario['end']}")
    print(f"Number of vehicles: {len(scenario['obstacles'])}")

    # Basic scenario checks
    check(scenario["scenario_id"] == "scenario_0001", "Incorrect scenario ID.")
    check(np.allclose(scenario["start"], [10.0, 10.0]), "Incorrect start position.")
    check(np.allclose(scenario["end"], [90.0, 90.0]), "Incorrect end position.")
    check("bounds_x" in scenario, "Missing bounds_x.")
    check("bounds_y" in scenario, "Missing bounds_y.")

    # Simulation checks
    check("simulation" in scenario, "Missing simulation configuration.")

    simulation = scenario["simulation"]

    check("dt" in simulation, "Missing simulation dt.")
    check(simulation["dt"] > 0.0, "Simulation dt must be positive.")
    check("reference_speed" in simulation, "Missing reference_speed.")
    check(simulation["reference_speed"] > 0.0, "Reference speed must be positive.")

    print(f"Simulation dt: {simulation['dt']} s")
    print(f"Reference speed: {simulation['reference_speed']}")

    # Vehicle checks
    check("vehicle" in scenario, "Missing vehicle configuration.")

    vehicle = scenario["vehicle"]

    check(vehicle["length"] > 0.0, "Vehicle length must be positive.")
    check(vehicle["width"] > 0.0, "Vehicle width must be positive.")
    check(vehicle["safety_distance"] > 0.0, "Safety distance must be positive.")

    # Obstacle checks
    check("obstacles" in scenario, "Missing obstacles.")
    check(len(scenario["obstacles"]) == 1, "Scenario should contain exactly one obstacle.")

    obstacle = scenario["obstacles"][0]

    print(
        f"Vehicle {obstacle['id']}: "
        f"position={obstacle['initial_position']}, "
        f"velocity={obstacle['velocity']}"
    )

    check("initial_position" in obstacle, "Obstacle is missing initial_position.")
    check("velocity" in obstacle, "Obstacle is missing velocity.")
    check("radius" in obstacle, "Obstacle is missing radius.")
    check(obstacle["radius"] > 0.0, "Obstacle radius must be positive.")

    print("\nAll scenario checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()