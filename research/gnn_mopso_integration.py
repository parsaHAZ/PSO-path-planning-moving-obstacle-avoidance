
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.gnn_behavior import InteractionGNN
from research.mopso import MOPSOPathPlanner
from research.gnn_prediction import GNNSafePredictor

NODE_DIM = 7
EDGE_DIM = 4


def load_gnn_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[GNNSafePredictor, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    predictor = GNNSafePredictor(
        checkpoint_path=checkpoint_path,
        model_class=InteractionGNN,
        device=device,
    )
    return predictor, checkpoint


def load_gnn(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    node_dim = int(checkpoint.get("node_dim", NODE_DIM))
    edge_dim = int(checkpoint.get("edge_dim", EDGE_DIM))
    hidden_dim = int(checkpoint.get("hidden_dim", 64))
    out_dim = int(checkpoint.get("out_dim", 3))

    if node_dim != NODE_DIM:
        raise ValueError(
            f"Checkpoint node_dim={node_dim}; expected {NODE_DIM}."
        )
    if edge_dim != EDGE_DIM:
        raise ValueError(
            f"Checkpoint edge_dim={edge_dim}; expected {EDGE_DIM}."
        )
    if out_dim != 3:
        raise ValueError(
            f"Checkpoint out_dim={out_dim}; expected 3."
        )

    gnn_model = InteractionGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    ).to(device)

    gnn_model.load_state_dict(checkpoint["model_state_dict"])
    gnn_model.eval()

    target_mean = np.asarray(
        checkpoint["target_mean"], dtype=np.float64
    )
    target_std = np.asarray(
        checkpoint["target_std"], dtype=np.float64
    )

    if target_mean.shape != (3,):
        raise ValueError(
            "target_mean must have shape (3,). "
            f"Got {target_mean.shape}."
        )
    if target_std.shape != (3,):
        raise ValueError(
            "target_std must have shape (3,). "
            f"Got {target_std.shape}."
        )
    if not np.isfinite(target_mean).all():
        raise ValueError("target_mean contains non-finite values.")
    if not np.isfinite(target_std).all():
        raise ValueError("target_std contains non-finite values.")
    if np.any(target_std <= 0):
        raise ValueError("target_std must contain positive values.")

    return gnn_model, target_mean, target_std


def predict_vehicle_behavior(
    model,
    graph,
    target_mean,
    target_std,
    device,
):
    x = graph["x"].to(device=device, dtype=torch.float32)
    edge_index = graph["edge_index"].to(
        device=device, dtype=torch.long
    )
    edge_attr = graph["edge_attr"].to(
        device=device, dtype=torch.float32
    )

    model.eval()
    with torch.no_grad():
        output = model(x, edge_index, edge_attr)

    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("GNN returned an empty tuple/list.")
        prediction = output[0]
    else:
        prediction = output

    if not torch.is_tensor(prediction):
        prediction = torch.as_tensor(
            prediction,
            dtype=torch.float32,
            device=device,
        )

    prediction = prediction.to(device=device, dtype=torch.float32)

    if prediction.ndim != 2:
        raise ValueError(
            "GNN prediction must have shape [N, 3]. "
            f"Got {tuple(prediction.shape)}."
        )
    if prediction.shape[0] != x.shape[0]:
        raise ValueError(
            f"Prediction/node mismatch: {prediction.shape[0]} "
            f"predictions for {x.shape[0]} nodes."
        )
    if prediction.shape[1] != 3:
        raise ValueError(
            f"Expected 3 predicted targets; got {prediction.shape[1]}."
        )

    target_mean_t = torch.as_tensor(
        target_mean, dtype=torch.float32, device=device
    ).reshape(-1)
    target_std_t = torch.as_tensor(
        target_std, dtype=torch.float32, device=device
    ).reshape(-1)

    if target_mean_t.numel() != 3 or target_std_t.numel() != 3:
        raise ValueError(
            "target_mean and target_std must each contain 3 values."
        )
    if not torch.isfinite(target_mean_t).all():
        raise ValueError("target_mean contains non-finite values.")
    if (
        not torch.isfinite(target_std_t).all()
        or torch.any(target_std_t <= 0)
    ):
        raise ValueError(
            "target_std must be finite and strictly positive."
        )

    prediction = (
        prediction * target_std_t.view(1, 3)
        + target_mean_t.view(1, 3)
    )

    node_features = x.detach().cpu().numpy().astype(np.float64)
    predictions = prediction.detach().cpu().numpy().astype(np.float64)

    if node_features.ndim != 2 or node_features.shape[1] != NODE_DIM:
        raise ValueError(
            f"Expected node features [N, {NODE_DIM}], "
            f"got {node_features.shape}."
        )
    if predictions.shape != (node_features.shape[0], 3):
        raise ValueError(
            "Final prediction shape mismatch: "
            f"{predictions.shape} vs {(node_features.shape[0], 3)}."
        )
    if not np.isfinite(node_features).all():
        raise ValueError("Node features contain non-finite values.")
    if not np.isfinite(predictions).all():
        raise ValueError("GNN predictions contain non-finite values.")

    return node_features, predictions


def _validate_node_features(node_features: np.ndarray) -> np.ndarray:
    features = np.asarray(node_features, dtype=np.float64)

    if features.ndim != 2:
        raise ValueError(
            f"node_features must be 2-D; got {features.shape}."
        )
    if features.shape[1] < 3:
        raise ValueError(
            "node_features must contain x, y, and speed."
        )
    if not np.isfinite(features[:, :3]).all():
        raise ValueError(
            "Node positions and speeds must be finite."
        )

    return features


def _vehicle_heading_direction(
    node_features: np.ndarray,
    node_index: int,
) -> np.ndarray:
    """Return the unit heading vector [cos(theta), sin(theta)].

    Node feature convention:
        index 5 = heading_sin
        index 6 = heading_cos
    """
    features = np.asarray(node_features, dtype=np.float64)

    if not 0 <= node_index < len(features):
        raise IndexError(
            f"node_index={node_index} is outside "
            f"[0, {len(features) - 1}]."
        )

    if features.shape[1] >= 7:
        heading_sin = float(features[node_index, 5])
        heading_cos = float(features[node_index, 6])

        direction = np.array(
            [heading_cos, heading_sin], dtype=np.float64
        )
        norm = float(np.linalg.norm(direction))

        if np.isfinite(norm) and norm > 1e-8:
            return direction / norm

    # Explicit fallback for absent or invalid heading information.
    return np.array([1.0, 0.0], dtype=np.float64)


def _get_track_ids(
    graph: dict[str, Any],
    node_count: int,
) -> np.ndarray:
    track_ids = graph.get("track_id")

    if track_ids is None:
        return np.arange(node_count)

    track_ids = np.asarray(track_ids)

    if len(track_ids) != node_count:
        raise ValueError(
            f"track_id count ({len(track_ids)}) does not match "
            f"node count ({node_count})."
        )

    return track_ids


def _track_id_value(track_ids: np.ndarray, node_index: int) -> int:
    try:
        return int(track_ids[node_index])
    except (TypeError, ValueError, IndexError, OverflowError):
        return int(node_index)


def build_observed_obstacles(
    graph: dict[str, Any],
    node_features: np.ndarray,
    ego_index: int,
    max_obstacles: int = 30,
    obstacle_radius: float = 2.5,
    min_distance: float = 2.0,
    max_distance: float = 40.0,
) -> list[dict[str, Any]]:
    features = _validate_node_features(node_features)

    if not 0 <= ego_index < len(features):
        raise IndexError(
            f"ego_index={ego_index} is outside "
            f"[0, {len(features) - 1}]."
        )
    if max_obstacles < 0:
        raise ValueError("max_obstacles must be non-negative.")
    if min_distance < 0 or max_distance < min_distance:
        raise ValueError(
            "Require 0 <= min_distance <= max_distance."
        )
    if obstacle_radius < 0:
        raise ValueError("obstacle_radius must be non-negative.")

    ego_position = features[ego_index, :2]
    track_ids = _get_track_ids(graph, len(features))
    obstacles: list[dict[str, Any]] = []

    for node_index in range(len(features)):
        if node_index == ego_index:
            continue

        position = features[node_index, :2].copy()
        distance = float(np.linalg.norm(position - ego_position))

        if distance < min_distance or distance > max_distance:
            continue

        speed = max(0.0, float(features[node_index, 2]))
        direction = _vehicle_heading_direction(features, node_index)
        velocity = direction * speed

        obstacles.append({
            "track_id": _track_id_value(track_ids, node_index),
            "initial_position": position,
            "velocity": velocity,
            "radius": float(obstacle_radius),
            "distance": distance,
        })

    obstacles.sort(key=lambda item: item["distance"])
    return obstacles[:max_obstacles]


def build_gnn_obstacles(
    graph: dict[str, Any],
    node_features: np.ndarray,
    predictions: np.ndarray,
    ego_index: int,
    max_obstacles: int = 30,
    obstacle_radius: float = 2.5,
    min_distance: float = 2.0,
    max_distance: float = 40.0,
    use_acceleration: bool = False,
) -> list[dict[str, Any]]:
    features = _validate_node_features(node_features)
    predictions = np.asarray(predictions, dtype=np.float64)

    if predictions.ndim != 2 or predictions.shape != (len(features), 3):
        raise ValueError(
            "predictions must have shape [N, 3] matching node_features; "
            f"got {predictions.shape} for {len(features)} nodes."
        )
    if not 0 <= ego_index < len(features):
        raise IndexError(
            f"ego_index={ego_index} is outside "
            f"[0, {len(features) - 1}]."
        )
    if max_obstacles < 0:
        raise ValueError("max_obstacles must be non-negative.")
    if min_distance < 0 or max_distance < min_distance:
        raise ValueError(
            "Require 0 <= min_distance <= max_distance."
        )
    if obstacle_radius < 0:
        raise ValueError("obstacle_radius must be non-negative.")

    ego_position = features[ego_index, :2]
    track_ids = _get_track_ids(graph, len(features))
    obstacles: list[dict[str, Any]] = []

    for node_index in range(len(features)):
        if node_index == ego_index:
            continue

        position = features[node_index, :2].copy()
        distance = float(np.linalg.norm(position - ego_position))

        if distance < min_distance or distance > max_distance:
            continue

        current_speed = float(features[node_index, 2])
        predicted_speed = float(predictions[node_index, 0])
        predicted_lon_acc = float(predictions[node_index, 1])
        predicted_lat_acc = float(predictions[node_index, 2])

        if not np.isfinite(predicted_speed):
            predicted_speed = current_speed
        if not np.isfinite(predicted_lon_acc):
            predicted_lon_acc = 0.0
        if not np.isfinite(predicted_lat_acc):
            predicted_lat_acc = 0.0

        predicted_speed = max(0.0, predicted_speed)
        direction = _vehicle_heading_direction(features, node_index)
        velocity = direction * predicted_speed

        obstacle: dict[str, Any] = {
            "track_id": _track_id_value(track_ids, node_index),
            "initial_position": position,
            "velocity": velocity,
            "radius": float(obstacle_radius),
            "distance": distance,
        }

        if use_acceleration:
            forward = direction
            lateral = np.array(
                [-forward[1], forward[0]], dtype=np.float64
            )
            acceleration = (
                predicted_lon_acc * forward
                + predicted_lat_acc * lateral
            )
            obstacle["acceleration"] = acceleration

        obstacles.append(obstacle)

    obstacles.sort(key=lambda item: item["distance"])
    return obstacles[:max_obstacles]


def create_planning_scenario(
    node_features: np.ndarray,
    ego_index: int,
    obstacles: list[dict[str, Any]],
    goal_distance: float = 30.0,
    safety_distance: float = 8.0,
) -> dict[str, Any]:
    features = _validate_node_features(node_features)

    if not 0 <= ego_index < len(features):
        raise IndexError(
            f"ego_index={ego_index} is outside "
            f"[0, {len(features) - 1}]."
        )
    if goal_distance <= 0:
        raise ValueError("goal_distance must be positive.")
    if safety_distance <= 0:
        raise ValueError("safety_distance must be positive.")

    start = features[ego_index, :2].copy()
    speed = max(0.0, float(features[ego_index, 2]))
    direction = _vehicle_heading_direction(features, ego_index)
    goal = start + direction * float(goal_distance)

    if obstacles:
        obstacle_positions = np.asarray(
            [obstacle["initial_position"] for obstacle in obstacles],
            dtype=np.float64,
        )

        if (
            obstacle_positions.ndim != 2
            or obstacle_positions.shape[1] != 2
            or not np.isfinite(obstacle_positions).all()
        ):
            raise ValueError(
                "Each obstacle must have a finite 2-D initial_position."
            )

        min_x = float(min(
            np.min(obstacle_positions[:, 0]), start[0], goal[0]
        ))
        max_x = float(max(
            np.max(obstacle_positions[:, 0]), start[0], goal[0]
        ))
        min_y = float(min(
            np.min(obstacle_positions[:, 1]), start[1], goal[1]
        ))
        max_y = float(max(
            np.max(obstacle_positions[:, 1]), start[1], goal[1]
        ))
    else:
        min_x = float(min(start[0], goal[0]))
        max_x = float(max(start[0], goal[0]))
        min_y = float(min(start[1], goal[1]))
        max_y = float(max(start[1], goal[1]))

    bounds_x = (
        min_x - float(safety_distance),
        max_x + float(safety_distance),
    )
    bounds_y = (
        min_y - float(safety_distance),
        max_y + float(safety_distance),
    )

    return {
        "start": start,
        "end": goal,
        "bounds_x": bounds_x,
        "bounds_y": bounds_y,
        "obstacles": obstacles,
        "reference_speed": speed if speed > 1e-8 else 1.0,
        "safety_distance": float(safety_distance),
    }


def load_graph(
    graph_path: str | Path,
    graph_index: int,
) -> dict[str, Any]:
    graph_path = Path(graph_path)
    data = torch.load(
        graph_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(data, (list, tuple)):
        raise TypeError(
            "Graph file must contain a list/tuple of graph snapshots. "
            f"Got {type(data).__name__}."
        )
    if not 0 <= graph_index < len(data):
        raise IndexError(
            f"graph_index={graph_index} is outside "
            f"[0, {len(data) - 1}]."
        )

    graph = data[graph_index]

    if not isinstance(graph, dict):
        raise TypeError(
            "Graph snapshot must be a dictionary. "
            f"Got {type(graph).__name__}."
        )

    required_keys = {"x", "edge_index", "edge_attr", "y"}
    missing = required_keys.difference(graph.keys())

    if missing:
        raise KeyError(
            f"Graph is missing required keys: {sorted(missing)}"
        )

    return graph


class GNNMOPSOPathPlanner(MOPSOPathPlanner):
    def __init__(
        self,
        start,
        end,
        bounds_x,
        bounds_y,
        scenario,
        n_waypoints=4,
        n_particles=100,
        max_iterations=50,
        inertia_weight=0.5,
        cognitive_coefficient=1.5,
        social_coefficient=1.5,
        archive_size=20,
        seed=42,
        use_acceleration=False,
        preference_model=None,
        preference_guidance_strength=0.35,
    ):
        self.use_acceleration = bool(use_acceleration)

        super().__init__(
            start=start,
            end=end,
            bounds_x=bounds_x,
            bounds_y=bounds_y,
            scenario=scenario,
            n_waypoints=n_waypoints,
            n_particles=n_particles,
            max_iterations=max_iterations,
            inertia_weight=inertia_weight,
            cognitive_coefficient=cognitive_coefficient,
            social_coefficient=social_coefficient,
            archive_size=archive_size,
            seed=seed,
            preference_model=preference_model,
            preference_guidance_strength=preference_guidance_strength,
        )

    def _moving_obstacle_position(self, obstacle, t):
        if "initial_position" in obstacle:
            initial_position = np.asarray(
                obstacle["initial_position"], dtype=np.float64
            )
        elif "position" in obstacle:
            initial_position = np.asarray(
                obstacle["position"], dtype=np.float64
            )
        else:
            raise KeyError(
                "Obstacle must contain 'initial_position' or 'position'."
            )

        velocity = np.asarray(
            obstacle.get("velocity", [0.0, 0.0]),
            dtype=np.float64,
        )

        if initial_position.shape != (2,) or velocity.shape != (2,):
            raise ValueError(
                "Obstacle position and velocity must each have shape (2,)."
            )

        if self.use_acceleration:
            acceleration = np.asarray(
                obstacle.get("acceleration", [0.0, 0.0]),
                dtype=np.float64,
            )
        else:
            acceleration = np.zeros(2, dtype=np.float64)

        if acceleration.shape != (2,):
            raise ValueError("Obstacle acceleration must have shape (2,).")

        t = float(t)

        return (
            initial_position
            + velocity * t
            + 0.5 * acceleration * (t ** 2)
        )
