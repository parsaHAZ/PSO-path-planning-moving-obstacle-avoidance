from data.scenarios.scenario_0001 import create_scenario


def main():
    scenario = create_scenario()
    print()
    print("=" * 50)
    print("SCENARIO TEST")
    print("=" * 50)
    print(f"Scenario ID: {scenario['scenario_id']}")
    print(f"Start: {scenario['start']}")
    print(f"End: {scenario['end']}")
    print(f"Number of vehicles: {len(scenario['obstacles'])}")
    for vehicle in scenario["obstacles"]:
        print(
            f"Vehicle {vehicle['id']}: "
            f"position={vehicle['initial_position']}, "
            f"velocity={vehicle['velocity']}"
        )
    print()
    print("Scenario loaded successfully.")

if __name__ == "__main__":
    main()