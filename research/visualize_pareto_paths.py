import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.scenarios.scenario_0001 import create_scenario
from research.mopso import generate_pareto_candidates


def draw_obstacle(ax, position, radius, linestyle, label,):
    circle = plt.Circle(position, radius, fill=False, linewidth=2, linestyle=linestyle, label=label,)
    ax.add_patch(circle)


def calculate_visualization_horizon(scenario):
    """
    Calculate a common physical-time horizon for visualization.

    The horizon is based on the straight-line distance from the
    vehicle start position to the goal at the scenario reference speed.
    """
    start = np.asarray( scenario["start"], dtype=float,)
    end = np.asarray( scenario["end"], dtype=float,)
    reference_speed = float(scenario["simulation"]["reference_speed"])
    distance = np.linalg.norm(end - start)
    if reference_speed <= 0:
        raise ValueError("Reference speed must be greater than zero.")

    return distance / reference_speed


def main():
    scenario = create_scenario(seed=42)
    candidates = generate_pareto_candidates(scenario=scenario, n_particles=100, max_iterations=100, archive_size=20, seed=42,)
    visualization_time = calculate_visualization_horizon(scenario)
    fig, ax = plt.subplots(figsize=(10, 9))

    # ------------------------------------------------------------
    # Start / goal
    # ------------------------------------------------------------

    start = scenario["start"]
    end = scenario["end"]
    ax.scatter(start[0], start[1], s=120, marker="o", label="Start", zorder=5,)
    ax.scatter(end[0], end[1], s=160, marker="*", label="Goal", zorder=5,)
    # ------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------
    for obstacle in scenario["obstacles"]:
        initial_position = np.asarray(obstacle["initial_position"], dtype=float,)
        velocity = np.asarray(obstacle["velocity"], dtype=float,)
        radius = float(obstacle.get("radius",0.0,))
        draw_obstacle(ax, initial_position, radius, "-", "Obstacle start",)
        final_position = (initial_position + velocity * visualization_time)
        draw_obstacle(ax, final_position, radius, "--", "Obstacle position at visualization horizon",)

    # ------------------------------------------------------------
    # Pareto trajectories
    # ------------------------------------------------------------
    for candidate in candidates:
        x, y = candidate["path"]
        safety, efficiency, comfort = (candidate["objectives"])
        label = (f"{candidate['candidate_id']} (S={safety:.3f}, E={efficiency:.4f}, C={comfort:.4f})")
        ax.plot(x, y, linewidth=1.5, alpha=0.65, label=label)
    # ------------------------------------------------------------
    # Figure configuration
    # ------------------------------------------------------------
    ax.set_xlim(scenario["bounds_x"])
    ax.set_ylim(scenario["bounds_y"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("MOPSO Pareto-Optimal Trajectories")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()