import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.scenarios.scenario_0001 import create_scenario
from research.mopso import generate_pareto_candidates


def print_objective_summary(candidates):
    """
    Print numerical information about the MOPSO Pareto archive.

    Objective order:
        0 = safety
        1 = efficiency
        2 = comfort
    """

    objectives = np.array(
        [candidate["objectives"] for candidate in candidates],
        dtype=float,
    )

    safety = objectives[:, 0]
    efficiency = objectives[:, 1]
    comfort = objectives[:, 2]

    print("\n" + "=" * 90)
    print("MOPSO PARETO FRONT - NUMERICAL OUTPUT")
    print("=" * 90)

    print(f"\nNumber of Pareto solutions: {len(candidates)}")

    print("\n" + "-" * 90)
    print(
        f"{'Candidate':<14}"
        f"{'Safety':>16}"
        f"{'Efficiency':>18}"
        f"{'Comfort':>16}"
    )
    print("-" * 90)

    for candidate in candidates:
        obj = candidate["objectives"]

        print(
            f"{candidate['candidate_id']:<14}"
            f"{obj[0]:>16.6f}"
            f"{obj[1]:>18.6f}"
            f"{obj[2]:>16.6f}"
        )

    print("-" * 90)

    print("\nOBJECTIVE RANGES")
    print("-" * 90)

    print(
        f"Safety:      "
        f"min={safety.min():.6f} | "
        f"max={safety.max():.6f} | "
        f"range={safety.max() - safety.min():.6f}"
    )

    print(
        f"Efficiency:  "
        f"min={efficiency.min():.6f} | "
        f"max={efficiency.max():.6f} | "
        f"range={efficiency.max() - efficiency.min():.6f}"
    )

    print(
        f"Comfort:     "
        f"min={comfort.min():.6f} | "
        f"max={comfort.max():.6f} | "
        f"range={comfort.max() - comfort.min():.6f}"
    )

    print("\n" + "=" * 90)
    print("EXTREME SOLUTIONS")
    print("=" * 90)

    safety_best = candidates[int(np.argmin(safety))]
    safety_worst = candidates[int(np.argmax(safety))]

    efficiency_best = candidates[int(np.argmin(efficiency))]
    efficiency_worst = candidates[int(np.argmax(efficiency))]

    comfort_best = candidates[int(np.argmin(comfort))]
    comfort_worst = candidates[int(np.argmax(comfort))]

    print(
        "\nLowest safety objective:\n"
        f"  {safety_best['candidate_id']} | "
        f"safety={safety_best['objectives'][0]:.6f}"
    )

    print(
        "\nHighest safety objective:\n"
        f"  {safety_worst['candidate_id']} | "
        f"safety={safety_worst['objectives'][0]:.6f}"
    )

    print(
        "\nLowest efficiency objective:\n"
        f"  {efficiency_best['candidate_id']} | "
        f"efficiency={efficiency_best['objectives'][1]:.6f}"
    )

    print(
        "\nHighest efficiency objective:\n"
        f"  {efficiency_worst['candidate_id']} | "
        f"efficiency={efficiency_worst['objectives'][1]:.6f}"
    )

    print(
        "\nLowest comfort objective:\n"
        f"  {comfort_best['candidate_id']} | "
        f"comfort={comfort_best['objectives'][2]:.6f}"
    )

    print(
        "\nHighest comfort objective:\n"
        f"  {comfort_worst['candidate_id']} | "
        f"comfort={comfort_worst['objectives'][2]:.6f}"
    )

    print("\n" + "=" * 90)
    print("TRADE-OFF CHECK")
    print("=" * 90)

    if len(candidates) > 1:
        corr_se = np.corrcoef(safety, efficiency)[0, 1]
        corr_sc = np.corrcoef(safety, comfort)[0, 1]
        corr_ec = np.corrcoef(efficiency, comfort)[0, 1]

        print(
            f"\nSafety vs Efficiency correlation: "
            f"{corr_se:.6f}"
        )

        print(
            f"Safety vs Comfort correlation:    "
            f"{corr_sc:.6f}"
        )

        print(
            f"Efficiency vs Comfort correlation:"
            f"{corr_ec:.6f}"
        )

    print("\n" + "=" * 90)
    print("DOMINANCE CHECK")
    print("=" * 90)

    def dominates(a, b):
        """
        Minimization dominance:
        a dominates b if a is no worse in all objectives
        and strictly better in at least one.
        """
        return (
            np.all(a <= b)
            and np.any(a < b)
        )

    dominated_pairs = []

    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i == j:
                continue

            if dominates(
                candidates[i]["objectives"],
                candidates[j]["objectives"],
            ):
                dominated_pairs.append(
                    (
                        candidates[i]["candidate_id"],
                        candidates[j]["candidate_id"],
                    )
                )

    if dominated_pairs:
        print(
            f"\nWARNING: {len(dominated_pairs)} "
            "dominance violations found."
        )

        for candidate_a, candidate_b in dominated_pairs[:20]:
            print(
                f"  {candidate_a} dominates {candidate_b}"
            )
    else:
        print(
            "\nAll archived solutions are mutually non-dominated."
        )

    print("\n" + "=" * 90)
    print("PARETO DIVERSITY")
    print("=" * 90)

    print(
        f"\nSafety standard deviation:     "
        f"{np.std(safety):.6f}"
    )

    print(
        f"Efficiency standard deviation: "
        f"{np.std(efficiency):.6f}"
    )

    print(
        f"Comfort standard deviation:    "
        f"{np.std(comfort):.6f}"
    )

    print("\n" + "=" * 90)
    print("RUN COMPLETE")
    print("=" * 90)


def main():
    scenario = create_scenario(seed=42)

    candidates = generate_pareto_candidates(
        scenario=scenario,
        n_particles=100,
        max_iterations=100,
        archive_size=30,
        seed=42,
    )

    print_objective_summary(candidates)


if __name__ == "__main__":
    main()