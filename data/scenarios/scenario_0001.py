import numpy as np


def create_scenario(seed=42):

    return {
        "scenario_id": "scenario_0001",
        "seed": int(seed),
        "start": np.array([10.0, 10.0], dtype=float,),
        "end": np.array([90.0, 90.0], dtype=float,),
        "bounds_x": (0.0, 100.0,),
        "bounds_y": (0.0, 100.0,),
        "simulation": {
            "dt": 0.1,
            "reference_speed": 10.0,
        },
        "vehicle": {
            "length": 4.5,
            "width": 2.0,
            "safety_distance": 8.0,
        },
        "obstacles": [
            {
                "id": "vehicle_001",

                "initial_position": np.array([45.0, 45.0], dtype=float,),
                "velocity": np.array([0.0, 0.15], dtype=float,),
                "radius": 5.0,
            }
        ],
        "metadata": {
            "scenario_type": "single_moving_vehicle",
            "traffic_density": "low",
            "driver_behavior": "predictable",
        },
    }