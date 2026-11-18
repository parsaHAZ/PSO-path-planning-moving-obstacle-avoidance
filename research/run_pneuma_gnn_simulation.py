"""
Run the trained pNEUMA interaction GNN inside a live moving-traffic / PSO demo.

Run from the repository root:
    D:\\Python310\\python.exe research\\run_pneuma_gnn_simulation.py

Required files:
    artifacts\\human_driving_validated_80ms\\best_interaction_gnn.pt
    artifacts\\human_driving_validated_80ms\\normalization.pt
    data\\pneuma\\graphs_fixed_80ms\\20181101_d9_0800_0830.pt

The GNN predicts an 80 ms next-state target. The script autoregressively
rolls its predictions forward to visualize traffic and supplies the forecast
to the existing PSO planner.

Important:
- The 4-second autoregressive forecast is not a validated long-horizon model.
- Soft collision penalties help PSO rank candidate paths, but do not guarantee
  a collision-free route.
- The AV's animation speed is illustrative, not calibrated vehicle dynamics.
"""

from pathlib import Path
import math
import sys
import warnings

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pso import PSOPathPlanner


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARTIFACTS = ROOT / "artifacts" / "human_driving_validated_80ms"
GRAPH_FILE = (
    ROOT / "data" / "pneuma" / "graphs_fixed_80ms"
    / "20181101_d9_0800_0830.pt"
)
CHECKPOINT = ARTIFACTS / "best_interaction_gnn.pt"
NORMALIZATION = ARTIFACTS / "normalization.pt"

DT = 0.080
EDGE_RADIUS_M = 40.0
MAX_VEHICLES = 80
ROLLOUT_STEPS = 50
SIM_FRAMES = 250
REPLAN_EVERY = 12

VEHICLE_RADIUS = 2.0
AV_RADIUS = 1.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Exact GNN architecture from validate_and_train_pneuma_80ms.py
# ---------------------------------------------------------------------------

class InteractionGNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.message = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.update = nn.GRUCell(hidden, hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.node_encoder(x)

        if edge_index.shape[1] == 0:
            return self.head(h)

        src, dst = edge_index
        e = self.edge_encoder(edge_attr)

        msg = self.message(
            torch.cat([h[dst], h[src], e], dim=1)
        )

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)

        degree = torch.zeros(
            (h.shape[0], 1),
            dtype=h.dtype,
            device=h.device,
        )

        degree.index_add_(
            0,
            dst,
            torch.ones(
                (dst.numel(), 1),
                dtype=h.dtype,
                device=h.device,
            ),
        )

        agg = agg / degree.clamp_min(1.0)
        h = self.update(agg, h)

        return self.head(h)


# ---------------------------------------------------------------------------
# Loading model and saved graph data
# ---------------------------------------------------------------------------

def load_pt(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model():
    for path in (CHECKPOINT, NORMALIZATION, GRAPH_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required project file not found:\n  {path}\n"
                "Check that the training artifacts and D9 graph file exist."
            )

    checkpoint = load_pt(CHECKPOINT)
    normalization = load_pt(NORMALIZATION)

    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Unexpected checkpoint keys: {list(checkpoint.keys())}"
        )

    model = InteractionGNN().to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    required_stats = (
        "x_mean",
        "x_std",
        "y_mean",
        "y_std",
        "edge_mean",
        "edge_std",
    )

    stats = {}
    for key in required_stats:
        if key not in normalization:
            raise KeyError(
                f"Normalization file is missing key {key!r}"
            )

        stats[key] = torch.as_tensor(
            normalization[key],
            dtype=torch.float32,
            device=DEVICE,
        )

    print(f"Loaded checkpoint: {CHECKPOINT}")
    print(
        f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}; "
        f"device: {DEVICE}"
    )

    return model, stats


# ---------------------------------------------------------------------------
# Dynamic graph reconstruction
# ---------------------------------------------------------------------------

def build_edges(x):
    """
    Construct directed edges with the training convention:
    edge features are destination-minus-source relative features.
    """
    positions = x[:, :2]
    speeds = x[:, 2]

    # delta[dst, src] = position(dst) - position(src)
    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.linalg.norm(delta, axis=2)

    dst, src = np.where(
        (distance <= EDGE_RADIUS_M) &
        (distance > 1e-6)
    )

    if len(dst):
        edge_attr = np.column_stack(
            (
                delta[dst, src, 0],
                delta[dst, src, 1],
                distance[dst, src],
                speeds[dst] - speeds[src],
            )
        ).astype(np.float32)

        edge_index = np.vstack((src, dst)).astype(np.int64)

    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, 4), dtype=np.float32)

    return edge_index, edge_attr


@torch.no_grad()
def predict_next(model, stats, x):
    edge_index, edge_attr = build_edges(x)

    xt = torch.as_tensor(
        x,
        dtype=torch.float32,
        device=DEVICE,
    )

    edge_index_t = torch.as_tensor(
        edge_index,
        dtype=torch.long,
        device=DEVICE,
    )

    edge_attr_t = torch.as_tensor(
        edge_attr,
        dtype=torch.float32,
        device=DEVICE,
    )

    x_normalized = (
        (xt - stats["x_mean"]) / stats["x_std"]
    )

    if len(edge_attr):
        edge_normalized = (
            (edge_attr_t - stats["edge_mean"]) /
            stats["edge_std"]
        )
    else:
        edge_normalized = edge_attr_t

    prediction_normalized = model(
        x_normalized,
        edge_index_t,
        edge_normalized,
    )

    prediction = (
        prediction_normalized * stats["y_std"] +
        stats["y_mean"]
    )

    return prediction.cpu().numpy()


# ---------------------------------------------------------------------------
# Advance traffic by one 80 ms model step
# ---------------------------------------------------------------------------

def advance_scene(x, prediction, dt=DT):
    """
    State features:
      x, y, speed, longitudinal acceleration, lateral acceleration,
      heading_sin, heading_cos

    GNN outputs:
      predicted speed, longitudinal acceleration, lateral acceleration
    """
    updated = x.copy()

    speed = np.clip(prediction[:, 0], 0.0, 40.0)
    a_lon = np.clip(prediction[:, 1], -8.0, 5.0)
    a_lat = np.clip(prediction[:, 2], -4.0, 4.0)

    heading = np.arctan2(updated[:, 5], updated[:, 6])
    old_speed = np.clip(updated[:, 2], 0.0, 40.0)

    heading += (
        a_lat / np.maximum(old_speed, 1.0)
    ) * dt

    mean_speed = 0.5 * (old_speed + speed)

    updated[:, 0] += mean_speed * np.cos(heading) * dt
    updated[:, 1] += mean_speed * np.sin(heading) * dt

    updated[:, 2] = speed
    updated[:, 3] = a_lon
    updated[:, 4] = a_lat
    updated[:, 5] = np.sin(heading)
    updated[:, 6] = np.cos(heading)

    return updated


# ---------------------------------------------------------------------------
# Choose a real recorded D9 traffic scene
# ---------------------------------------------------------------------------

def choose_initial_scene():
    graphs = load_pt(GRAPH_FILE)

    if not isinstance(graphs, (list, tuple)):
        raise TypeError(
            f"Expected a list or tuple of graphs, got {type(graphs)}"
        )

    candidates = []

    for graph_index, graph in enumerate(graphs):
        if "x" not in graph:
            continue

        features = graph["x"]

        if not torch.is_tensor(features):
            features = torch.as_tensor(features)

        features = (
            features.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if features.ndim != 2 or features.shape[1] != 7:
            continue

        if len(features) < 4:
            continue

        if not np.isfinite(features).all():
            continue

        spread = np.linalg.norm(
            features[:, :2] - features[0, :2],
            axis=1,
        )

        if np.nanmax(spread) < 8.0:
            continue

        candidates.append(
            (abs(len(features) - 24), graph_index, features)
        )

        if len(candidates) >= 250:
            break

    if not candidates:
        raise RuntimeError(
            "No suitable finite D9 scene with at least four vehicles."
        )

    _, graph_index, state = min(
        candidates,
        key=lambda item: item[0],
    )

    if len(state) > MAX_VEHICLES:
        state = state[:MAX_VEHICLES].copy()

    distances = np.linalg.norm(
        state[:, None, :2] - state[None, :, :2],
        axis=2,
    )

    degree = (
        (distances <= EDGE_RADIUS_M) &
        (distances > 1e-6)
    ).sum(axis=1)

    human_index = int(np.argmax(degree))

    print(
        f"Using D9 graph index {graph_index}; "
        f"vehicles={len(state)}; "
        f"highlighted vehicle={human_index}"
    )
    print(
        "All vehicles are advanced by the trained GNN so that "
        "surrounding traffic remains interaction context."
    )

    return state, human_index


# ---------------------------------------------------------------------------
# PSO planner with GNN-predicted moving obstacles
# ---------------------------------------------------------------------------

class GNNForecastPSO(PSOPathPlanner):
    """
    Reuse the repository PSO planner.

    The path objective uses finite soft collision penalties. This avoids
    the original failure mode in which every initial particle receives
    infinite cost and PSO never initializes a global best solution.
    """

    def __init__(
        self,
        start,
        end,
        bounds_x,
        bounds_y,
        predicted_positions,
        seed=42,
    ):
        super().__init__(
            start=start,
            end=end,
            n_obstacles=0,
            max_iterations=22,
            bounds_x=bounds_x,
            bounds_y=bounds_y,
            scenario={
                "seed": seed,
                "obstacles": [],
            },
        )

        self.n_particles = 42
        self.n_waypoints = 4
        self.predicted_positions = predicted_positions

        self.prediction_horizon_s = (
            len(predicted_positions) - 1
        ) * DT

    def _spline_path(self, waypoints):
        """
        Override the base spline fitting with a fallback sequence.
        """
        from scipy.interpolate import splprep, splev

        points = np.vstack(
            [self.start, waypoints, self.end]
        ).T

        # Duplicate points can make splprep fail. The loop also retries
        # with less smoothing if FITPACK cannot fit a particular spline.
        for smoothing in (5.0, 2.0, 0.5, 0.0):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    tck, _ = splprep(points, s=smoothing)

                u = np.linspace(0.0, 1.0, 100)
                x_spline, y_spline = splev(u, tck)

                if (
                    np.isfinite(x_spline).all() and
                    np.isfinite(y_spline).all()
                ):
                    return x_spline, y_spline

            except Exception:
                continue

        return None, None

    def _is_valid_path(
        self,
        x_spline,
        y_spline,
        iteration=0,
        dynamic=True,
    ):
        """
        Return True when the sampled path has no predicted overlap.
        Used as a diagnostic; _path_cost uses soft penalties instead.
        """
        if self.predicted_positions is None:
            return True

        indices = np.linspace(
            0,
            len(self.predicted_positions) - 1,
            len(x_spline),
        ).round().astype(int)

        minimum_distance = VEHICLE_RADIUS + AV_RADIUS

        for i, (px, py) in enumerate(
            zip(x_spline, y_spline)
        ):
            traffic = self.predicted_positions[indices[i]]
            distances = np.linalg.norm(
                traffic - np.array([px, py]),
                axis=1,
            )

            if np.any(distances < minimum_distance):
                return False

        return True

    def _path_cost(
        self,
        x_spline,
        y_spline,
        iteration=0,
    ):
        """
        Score path length plus soft penalties for proximity to traffic.
        """
        if (
            x_spline is None or
            y_spline is None or
            not np.isfinite(x_spline).all() or
            not np.isfinite(y_spline).all()
        ):
            return 1e12

        path_length = float(
            np.sum(
                np.hypot(
                    np.diff(x_spline),
                    np.diff(y_spline),
                )
            )
        )

        time_indices = np.linspace(
            0,
            len(self.predicted_positions) - 1,
            len(x_spline),
        ).round().astype(int)

        clearance = VEHICLE_RADIUS + AV_RADIUS
        penalty = 0.0

        for i, (px, py) in enumerate(
            zip(x_spline, y_spline)
        ):
            traffic = self.predicted_positions[time_indices[i]]

            distances = np.linalg.norm(
                traffic - np.array([px, py], dtype=float),
                axis=1,
            )

            # Penalty begins before the nominal clearance boundary.
            deficit = np.maximum(
                clearance + 1.0 - distances,
                0.0,
            )

            # Stronger penalty for actual overlap.
            penetration = np.maximum(
                clearance - distances,
                0.0,
            )

            penalty += 20.0 * float(
                np.sum(deficit ** 2)
            )

            penalty += 200.0 * float(
                np.sum(penetration ** 2)
            )

        return path_length + penalty


# ---------------------------------------------------------------------------
# GNN traffic forecast and AV path planning
# ---------------------------------------------------------------------------

def rollout_positions(
    model,
    stats,
    initial_state,
    steps=ROLLOUT_STEPS,
):
    state = initial_state.copy()
    positions = [state[:, :2].copy()]

    for _ in range(steps):
        prediction = predict_next(model, stats, state)
        state = advance_scene(state, prediction)
        positions.append(state[:, :2].copy())

    return np.stack(positions, axis=0)


def plan_av(
    model,
    stats,
    traffic_state,
    av_position,
    destination,
    frame,
):
    forecast = rollout_positions(
        model,
        stats,
        traffic_state,
    )

    margin = 12.0

    bounds_x = (
        min(av_position[0], destination[0]) - margin,
        max(av_position[0], destination[0]) + margin,
    )

    bounds_y = (
        min(av_position[1], destination[1]) - margin,
        max(av_position[1], destination[1]) + margin,
    )

    planner = GNNForecastPSO(
        start=av_position,
        end=destination,
        bounds_x=bounds_x,
        bounds_y=bounds_y,
        predicted_positions=forecast,
        seed=100 + frame,
    )

    result, cost, iteration, _ = planner.run()

    # The repository's planner returns:
    # ((x_spline, y_spline), cost, iteration, convergence)
    # or (None, inf, -1, convergence) if no spline was produced.
    if (
        result is None or
        result[0] is None or
        result[1] is None
    ):
        print(
            f"PSO returned no path at frame {frame}; "
            "this planning attempt failed."
        )
        return None

    path = np.column_stack(result)

    # Sampled minimum center distance to traffic. This is a diagnostic,
    # not a continuous-time collision proof.
    sample_indices = np.linspace(
        0,
        len(forecast) - 1,
        len(path),
    ).round().astype(int)

    minimum_distance = float("inf")

    for i, point in enumerate(path):
        distances = np.linalg.norm(
            forecast[sample_indices[i]] - point,
            axis=1,
        )

        if len(distances):
            minimum_distance = min(
                minimum_distance,
                float(distances.min()),
            )

    if minimum_distance < VEHICLE_RADIUS + AV_RADIUS:
        print(
            "WARNING: selected path overlaps sampled traffic forecast; "
            f"minimum center distance={minimum_distance:.2f} m."
        )
    else:
        print(
            "Sampled forecast clearance: "
            f"minimum center distance={minimum_distance:.2f} m."
        )

    return path, cost, iteration, forecast


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def main():
    model, stats = load_model()
    state, human_index = choose_initial_scene()
    vehicle_count = len(state)

    human_position = state[human_index, :2].copy()

    heading = math.atan2(
        float(state[human_index, 5]),
        float(state[human_index, 6]),
    )

    direction = np.array(
        [math.cos(heading), math.sin(heading)]
    )

    normal = np.array(
        [-direction[1], direction[0]]
    )

    # Put the AV on a crossing route relative to the highlighted vehicle.
    av_position = (
        human_position -
        26.0 * direction +
        10.0 * normal
    )

    destination = (
        human_position +
        26.0 * direction -
        10.0 * normal
    )

    # Move the AV start farther away if it initially overlaps traffic.
    for _ in range(8):
        nearest_distance = np.min(
            np.linalg.norm(
                state[:, :2] - av_position,
                axis=1,
            )
        )

        if nearest_distance > VEHICLE_RADIUS + AV_RADIUS + 1.0:
            break

        av_position = av_position - 3.0 * normal

    av_position = av_position.astype(float)
    destination = destination.astype(float)

    print(
        "Planning initial AV path against a 4-second "
        "autoregressive GNN traffic forecast..."
    )

    planned = plan_av(
        model,
        stats,
        state,
        av_position,
        destination,
        frame=0,
    )

    if planned is None:
        print("Retrying with a wider crossing offset.")

        av_position = (
            human_position -
            30.0 * direction +
            16.0 * normal
        )

        destination = (
            human_position +
            30.0 * direction -
            16.0 * normal
        )

        av_position = av_position.astype(float)
        destination = destination.astype(float)

        planned = plan_av(
            model,
            stats,
            state,
            av_position,
            destination,
            frame=1,
        )

    if planned is None:
        raise RuntimeError(
            "PSO did not produce a spline for either initial attempt. "
            "Try another D9 scene or inspect the path-planner bounds."
        )

    av_path, av_cost, av_iteration, forecast = planned

    fig, ax = plt.subplots(figsize=(10, 8))

    traffic_scatter = ax.scatter(
        state[:, 0],
        state[:, 1],
        s=80,
        c="#9aa0a6",
        edgecolors="black",
        linewidths=0.4,
        label="Traffic advanced by GNN",
    )

    human_scatter = ax.scatter(
        [state[human_index, 0]],
        [state[human_index, 1]],
        s=150,
        c="#e53935",
        edgecolors="black",
        linewidths=0.8,
        label="Highlighted GNN-driven vehicle",
        zorder=5,
    )

    av_dot, = ax.plot(
        [],
        [],
        "o",
        color="#1565c0",
        markersize=10,
        label="AV",
    )

    av_trail, = ax.plot(
        [],
        [],
        "-",
        color="#1565c0",
        linewidth=2,
        label="AV traveled",
    )

    plan_line, = ax.plot(
        av_path[:, 0],
        av_path[:, 1],
        "--",
        color="#00a6a6",
        linewidth=1.8,
        label="PSO plan vs GNN forecast",
    )

    human_prediction_line, = ax.plot(
        forecast[:, human_index, 0],
        forecast[:, human_index, 1],
        ":",
        color="#e53935",
        linewidth=1.8,
        label="Highlighted vehicle GNN rollout",
    )

    vehicle_circles = []

    for i in range(vehicle_count):
        circle = Circle(
            state[i, :2],
            VEHICLE_RADIUS,
            fill=False,
            edgecolor="#666666",
            alpha=0.5,
            linewidth=0.7,
        )

        ax.add_patch(circle)
        vehicle_circles.append(circle)

    av_circle = Circle(
        av_position,
        AV_RADIUS,
        color="#1565c0",
        alpha=0.25,
    )

    ax.add_patch(av_circle)

    all_xy = np.vstack(
        [
            state[:, :2],
            av_path,
            forecast.reshape(-1, 2),
        ]
    )

    lower = np.nanmin(all_xy, axis=0) - 12.0
    upper = np.nanmax(all_xy, axis=0) + 12.0

    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    ax.set_xlabel("pNEUMA x (recorded graph coordinates)")
    ax.set_ylabel("pNEUMA y (recorded graph coordinates)")

    ax.legend(loc="upper left", fontsize=8)

    title = ax.set_title(
        "Trained pNEUMA Interaction GNN + PSO moving-obstacle simulation"
    )

    fig.tight_layout()

    history = [av_position.copy()]
    frame_counter = {"i": 0}

    current_plan = {
        "path": av_path,
        "forecast": forecast,
        "cost": av_cost,
        "iteration": av_iteration,
    }

    def update(_):
        nonlocal state, av_position

        frame = frame_counter["i"]

        # Replan against an updated GNN forecast every ~0.96 s.
        if frame > 0 and frame % REPLAN_EVERY == 0:
            new_plan = plan_av(
                model,
                stats,
                state,
                av_position,
                destination,
                frame,
            )

            if new_plan is not None:
                (
                    current_plan["path"],
                    current_plan["cost"],
                    current_plan["iteration"],
                    current_plan["forecast"],
                ) = new_plan

                plan_line.set_data(
                    current_plan["path"][:, 0],
                    current_plan["path"][:, 1],
                )

                human_prediction_line.set_data(
                    current_plan["forecast"][:, human_index, 0],
                    current_plan["forecast"][:, human_index, 1],
                )

        # Advance traffic using the model's next-step predictions.
        prediction = predict_next(model, stats, state)
        state = advance_scene(state, prediction)

        traffic_scatter.set_offsets(state[:, :2])

        human_scatter.set_offsets(
            state[[human_index], :2]
        )

        for i, circle in enumerate(vehicle_circles):
            circle.center = state[i, :2]

        # Move the AV along the currently selected PSO path.
        # This is an illustrative animation parameterization.
        path = current_plan["path"]

        progress = frame % max(1, len(path))
        av_position = path[progress].copy()

        history.append(av_position.copy())

        av_dot.set_data(
            [av_position[0]],
            [av_position[1]],
        )

        av_circle.center = av_position

        history_array = np.asarray(history)

        av_trail.set_data(
            history_array[:, 0],
            history_array[:, 1],
        )

        title.set_text(
            f"GNN-driven traffic + PSO AV | "
            f"t={frame * DT:.2f}s | "
            f"vehicles={vehicle_count} | "
            f"PSO cost={current_plan['cost']:.1f}"
        )

        frame_counter["i"] += 1

        return [
            traffic_scatter,
            human_scatter,
            av_dot,
            av_trail,
            plan_line,
            human_prediction_line,
            av_circle,
            *vehicle_circles,
        ]

    animation = FuncAnimation(
        fig,
        update,
        frames=SIM_FRAMES,
        interval=80,
        blit=False,
        repeat=False,
    )

    # Keep a reference to prevent garbage collection.
    fig._pneuma_animation = animation

    plt.show()


if __name__ == "__main__":
    main()