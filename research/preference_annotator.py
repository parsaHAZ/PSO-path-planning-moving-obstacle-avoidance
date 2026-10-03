import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pso import PSOPathPlanner
from data.scenarios.scenario_0001 import create_scenario

class PreferenceAnnotator:

    def __init__(self, dataset_path, output_path, n_candidates=5):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        with self.dataset_path.open("r", encoding="utf-8") as file:
            self.dataset = json.load(file)

        self.scenario = create_scenario()
        planner = PSOPathPlanner(
            start=self.scenario["start"],
            end=self.scenario["end"],
            n_obstacles=0,
            max_iterations=50,
            bounds_x=self.scenario["bounds_x"],
            bounds_y=self.scenario["bounds_y"],
            scenario=self.scenario,
        )
        self.candidates = planner.generate_candidates(n_candidates=n_candidates)
        self.candidate_lookup = {
            candidate["candidate_id"]: candidate
            for candidate in self.candidates
        }
        self.current_index = 0
        self.fig = None
        self.ax = None
        self.title_text = None
        self.button_a = None
        self.button_b = None
        self.button_tie = None
        self.button_skip = None

    def current_pair(self):
        return self.dataset[self.current_index]

    def get_candidate(self, candidate_id):
        return self.candidate_lookup[candidate_id]

    def draw_obstacle(self, ax, position, radius, color, label=None):
        circle = plt.Circle(
            position,
            radius,
            fill=False,
            linewidth=2.0,
            color=color,
            label=label,
        )
        ax.add_patch(circle)

    def draw_candidate(self, ax, candidate, label, linestyle):
        x, y = candidate["path"]
        ax.plot(x, y, linestyle=linestyle, linewidth=2.5, label=label,)

    def redraw(self):
        pair = self.current_pair()
        candidate_a = self.get_candidate(pair["candidate_a"])
        candidate_b = self.get_candidate(pair["candidate_b"])
        self.ax.clear()
        start = self.scenario["start"]
        end = self.scenario["end"]
        self.ax.scatter(start[0], start[1], s=100, marker="o", label="Start", zorder=5)
        self.ax.scatter(end[0], end[1], s=100, marker="*", label="Goal", zorder=5)
        self.draw_candidate(self.ax, candidate_a, "Candidate A", "-")
        self.draw_candidate(self.ax, candidate_b, "Candidate B", "--")
        for obstacle in self.scenario["obstacles"]:
            initial_position = obstacle["initial_position"]
            velocity = obstacle["velocity"]
            radius = float(obstacle.get("radius", 0.0))
            self.draw_obstacle(self.ax, initial_position, radius, "black", "Obstacle start")
            _, y_a = candidate_a["path"]
            final_step = len(y_a) - 1
            final_position = initial_position + velocity * final_step
            self.draw_obstacle(self.ax, final_position, radius, "gray", "Obstacle end")

        self.ax.set_xlim(self.scenario["bounds_x"])
        self.ax.set_ylim(self.scenario["bounds_y"])
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlabel("X position")
        self.ax.set_ylabel("Y position")
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc="upper left")
        comparison_number = self.current_index + 1
        total_comparisons = len(self.dataset)
        self.ax.set_title(
            f"Preference Comparison {comparison_number} / {total_comparisons}\n"
            f"{pair['candidate_a']}  vs  {pair['candidate_b']}"
        )

        self.fig.canvas.draw_idle()

    def save_dataset(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(self.dataset, file, indent=4)

    def record_preference(self, preference):
        pair = self.current_pair()
        pair["preferred_candidate"] = preference
        self.save_dataset()

        print(f"[{self.current_index + 1}/{len(self.dataset)}] "
            f"{pair['comparison_id']} -> {preference}"
        )

        self.next_comparison()
    def choose_a(self, event):
        pair = self.current_pair()
        self.record_preference(pair["candidate_a"])

    def choose_b(self, event):
        pair = self.current_pair()
        self.record_preference(pair["candidate_b"])

    def choose_tie(self, event):
        self.record_preference("tie")

    def skip(self, event):
        print(
            f"[{self.current_index + 1}/{len(self.dataset)}] "
            f"Skipped {self.current_pair()['comparison_id']}"
        )
        self.next_comparison()

    def next_comparison(self):
        if self.current_index >= len(self.dataset) - 1:
            print()
            print("All comparisons completed.")
            self.save_dataset()
            plt.close(self.fig)
            return

        self.current_index += 1
        self.redraw()
    
    def run(self):
        self.fig, self.ax = plt.subplots(figsize=(9, 8))
        plt.subplots_adjust(bottom=0.18)
        ax_button_a = plt.axes([0.12, 0.05, 0.18, 0.06])
        ax_button_b = plt.axes([0.32, 0.05, 0.18, 0.06])
        ax_button_tie = plt.axes([0.52, 0.05, 0.18, 0.06])
        ax_button_skip = plt.axes([0.72, 0.05, 0.15, 0.06])
        self.button_a = Button(ax_button_a, "Prefer A")
        self.button_b = Button(ax_button_b, "Prefer B")
        self.button_tie = Button(ax_button_tie, "Tie / Unsure")
        self.button_skip = Button(ax_button_skip, "Skip")
        self.button_a.on_clicked(self.choose_a)
        self.button_b.on_clicked(self.choose_b)
        self.button_tie.on_clicked(self.choose_tie)
        self.button_skip.on_clicked(self.skip)
        self.redraw()
        plt.show()