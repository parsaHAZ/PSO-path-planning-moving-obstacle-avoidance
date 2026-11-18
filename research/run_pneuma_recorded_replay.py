
"""
Independent simulated ego in recorded pNEUMA traffic.

Gray dots: recorded surrounding vehicles.
Yellow dot: independently simulated ego.
Green x: recorded ego position, reference only.
Orange diamond/dashed line: GNN nominal next position/rollout.
Cyan star/line: controller-selected next position/rollout.
Red dots/lines: approximate predicted obstacle futures.
Gold line: recent simulated ego path.

Important:
- The ego is initialized from the recorded track only once.
- The ego is never re-anchored to its recorded future positions.
- Surrounding vehicles follow recorded trajectories and do not react to the ego.
- Traffic-corridor following is a heuristic, not a road map or a learned lane model.

Run from repository root:
  D:\\Python310\\python.exe research\\run_pneuma_recorded_replay.py --day 5 --start-frame 5000 --frames 1200 --track-id 538 --save-metrics
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

ROOT = Path.cwd()
GRAPH_DIR = ROOT / "data" / "pneuma" / "graphs_fixed_80ms"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "gnn" / "20181101_multidrone.pt"

Y_MEAN = torch.tensor([3.89977551, .06114579, .0127314], dtype=torch.float32)
Y_STD = torch.tensor([4.28055429, .77070218, .36888441], dtype=torch.float32)


class InteractionGNN(nn.Module):
    """Checkpoint-compatible model; preserve state-dict parameter names."""

    def __init__(self, node_in=7, edge_in=4, hidden=64, out=3):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.update = nn.GRUCell(hidden, hidden)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out)
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        agg = torch.zeros_like(h)

        if edge_index.numel():
            src, dst = edge_index
            msg = self.message_mlp(torch.cat((h[dst], h[src], e), dim=1))
            agg.index_add_(0, dst, msg)
            deg = torch.zeros((len(h), 1), dtype=h.dtype, device=h.device)
            deg.index_add_(
                0, dst,
                torch.ones((len(dst), 1), dtype=h.dtype, device=h.device)
            )
            agg = agg / deg.clamp_min(1.0)

        return self.prediction_head(self.update(agg, h))


def get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def as_tensor(value, dtype=None):
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def graph_dict(graph):
    out = {
        key: get(graph, key)
        for key in ("x", "edge_index", "edge_attr", "track_id")
        if get(graph, key) is not None
    }
    missing = {"x", "edge_index", "edge_attr", "track_id"} - out.keys()
    if missing:
        raise KeyError(f"Graph missing {sorted(missing)}")

    out["x"] = as_tensor(out["x"], torch.float32).cpu()
    out["edge_index"] = as_tensor(out["edge_index"], torch.long).cpu()
    out["edge_attr"] = as_tensor(out["edge_attr"], torch.float32).cpu()
    out["track_id"] = as_tensor(out["track_id"]).reshape(-1).cpu()

    if out["x"].ndim != 2 or out["x"].shape[1] != 7:
        raise ValueError(f"Expected x [N,7], got {tuple(out['x'].shape)}")
    if out["edge_index"].ndim != 2 or out["edge_index"].shape[0] != 2:
        raise ValueError("edge_index must be [2,E]")
    if out["edge_attr"].shape != (out["edge_index"].shape[1], 4):
        raise ValueError("edge_attr must be [E,4]")
    if out["track_id"].numel() != out["x"].shape[0]:
        raise ValueError("track_id count mismatch")

    return out


def load_graphs(path):
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")

    if isinstance(raw, dict):
        for key in ("graphs", "frames", "data"):
            if isinstance(raw.get(key), (list, tuple)):
                raw = raw[key]
                break

    if not isinstance(raw, (list, tuple)) or not raw:
        raise TypeError(f"Expected nonempty graph list, got {type(raw).__name__}")

    return [graph_dict(graph) for graph in raw]


def load_model(path, device):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    state = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "gnn_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                state = checkpoint[key]
                break

    model = InteractionGNN().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    mean, std = Y_MEAN.clone(), Y_STD.clone()
    if isinstance(checkpoint, dict):
        containers = [checkpoint] + [
            checkpoint[key]
            for key in ("normalization", "norm", "normalizers", "stats")
            if isinstance(checkpoint.get(key), dict)
        ]
        for container in containers:
            for mean_key, std_key in (
                ("y_mean", "y_std"),
                ("target_mean", "target_std"),
                ("target_mu", "target_sigma"),
            ):
                if mean_key in container and std_key in container:
                    candidate_mean = as_tensor(
                        container[mean_key], torch.float32
                    ).reshape(-1)
                    candidate_std = as_tensor(
                        container[std_key], torch.float32
                    ).reshape(-1)
                    if candidate_mean.numel() == 3 and candidate_std.numel() == 3:
                        mean = candidate_mean
                        std = candidate_std.clamp_min(1e-8)
                        break

    print("Checkpoint loaded with strict=True.")
    print("Output target mean:", mean.tolist())
    print("Output target std: ", std.tolist())
    return model, mean, std


def edge_features(x, edge_index):
    if not edge_index.numel():
        return torch.empty((0, 4), dtype=x.dtype)

    src, dst = edge_index
    delta = x[dst, :2] - x[src, :2]
    distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(1e-6)
    relative_speed = (x[dst, 2] - x[src, 2])[:, None]
    return torch.cat((delta, distance[:, None], relative_speed), dim=1)


def radius_graph(x, radius):
    if len(x) < 2:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 4), dtype=x.dtype),
        )

    delta = x[:, None, :2] - x[None, :, :2]
    distance = torch.linalg.vector_norm(delta, dim=2)
    mask = (distance <= radius) & (distance > 0)
    mask.fill_diagonal_(False)
    dst, src = torch.nonzero(mask, as_tuple=True)
    edge_index = torch.stack((src, dst)).long()
    return edge_index, edge_features(x, edge_index)


def make_graph(graph, track_id, ego_state, radius):
    x = graph["x"].clone()
    ids = graph["track_id"].clone()
    hit = torch.nonzero(ids == track_id).reshape(-1)

    if hit.numel():
        ego_index = int(hit[0])
        x[ego_index] = torch.as_tensor(ego_state, dtype=torch.float32)
    else:
        ego_index = len(x)
        x = torch.cat((
            x,
            torch.as_tensor(ego_state, dtype=torch.float32).reshape(1, 7),
        ))
        ids = torch.cat((ids, torch.as_tensor([track_id], dtype=ids.dtype)))

    edge_index, edge_attr = radius_graph(x, radius)
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "track_id": ids,
    }, ego_index


def predict(model, graph, mean, std, device):
    with torch.inference_mode():
        output = model(
            graph["x"].to(device),
            graph["edge_index"].to(device),
            graph["edge_attr"].to(device),
        )
    return (output.cpu() * std + mean).numpy()


def heading(state):
    return math.atan2(float(state[5]), float(state[6]))


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def set_heading(state, angle):
    result = np.asarray(state, dtype=np.float32).copy()
    result[5] = math.sin(angle)
    result[6] = math.cos(angle)
    return result


def integrate(state, a_lon, a_lat, dt):
    """Integrate one bounded kinematic step."""
    s = np.asarray(state, dtype=np.float32).copy()
    v = max(0.0, float(s[2]))
    a_lon = float(np.clip(a_lon, -3.0, 2.0))
    a_lat = float(np.clip(a_lat, -1.5, 1.5))

    v_next = max(0.0, v + a_lon * dt)
    v_mid = 0.5 * (v + v_next)
    yaw_rate = float(np.clip(a_lat / max(v_mid, 1.0), -0.20, 0.20))
    h_mid = heading(s) + 0.5 * yaw_rate * dt

    s[0] += v_mid * math.cos(h_mid) * dt
    s[1] += v_mid * math.sin(h_mid) * dt
    s[2:5] = (v_next, a_lon, a_lat)
    return set_heading(s, wrap_angle(heading(state) + yaw_rate * dt))


def rollout(state, actions, times, step):
    """Roll out candidate [a_longitudinal, a_lateral] actions."""
    actions = np.asarray(actions, dtype=float)
    count = len(actions)
    pos = np.repeat(np.asarray(state[:2], float)[None, :], count, axis=0)
    speed = np.full(count, max(0.0, float(state[2])))
    angle = np.full(count, heading(state))
    result = np.empty((count, len(times), 2))
    previous = 0.0

    for j, target_time in enumerate(times):
        remaining = float(target_time - previous)
        while remaining > 1e-8:
            d = min(step, remaining)
            next_speed = np.maximum(
                0.0, speed + np.clip(actions[:, 0], -3.0, 2.0) * d
            )
            mean_speed = 0.5 * (speed + next_speed)
            yaw_rate = np.clip(
                np.clip(actions[:, 1], -1.5, 1.5)
                / np.maximum(mean_speed, 1.0),
                -0.20, 0.20,
            )
            mid_angle = angle + 0.5 * yaw_rate * d
            pos[:, 0] += mean_speed * np.cos(mid_angle) * d
            pos[:, 1] += mean_speed * np.sin(mid_angle) * d
            angle = (angle + yaw_rate * d + np.pi) % (2.0 * np.pi) - np.pi
            speed = next_speed
            remaining -= d

        result[:, j] = pos
        previous = float(target_time)

    return result


def obstacle_rollouts(states, predictions, times, step):
    """Approximate surrounding-vehicle futures using their GNN predictions."""
    if len(states) == 0:
        return np.empty((0, len(times), 2))

    states = np.asarray(states, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    count = len(states)
    pos = states[:, :2].copy()
    speed = np.maximum(0.0, states[:, 2])
    angle = np.arctan2(states[:, 5], states[:, 6])
    result = np.empty((count, len(times), 2))
    previous = 0.0

    for j, target_time in enumerate(times):
        remaining = float(target_time - previous)
        while remaining > 1e-8:
            d = min(step, remaining)
            next_speed = np.maximum(
                0.0, speed + np.clip(predictions[:, 1], -3.0, 2.0) * d
            )
            mean_speed = 0.5 * (speed + next_speed)
            yaw_rate = np.clip(
                np.clip(predictions[:, 2], -1.5, 1.5)
                / np.maximum(mean_speed, 1.0),
                -0.20, 0.20,
            )
            mid_angle = angle + 0.5 * yaw_rate * d
            pos[:, 0] += mean_speed * np.cos(mid_angle) * d
            pos[:, 1] += mean_speed * np.sin(mid_angle) * d
            angle = (angle + yaw_rate * d + np.pi) % (2.0 * np.pi) - np.pi
            speed = next_speed
            remaining -= d

        result[:, j] = pos
        previous = float(target_time)

    return result


def local_flow_heading(graph, ego_state, track_id, max_distance=30.0):
    """
    Estimate traffic direction around the simulated ego.

    The selected track's recorded heading is deliberately excluded: after the
    ego leaves its original location, that heading may no longer represent
    the traffic near the simulated ego.
    """
    x = graph["x"].numpy()
    ids = graph["track_id"].numpy()

    if len(x) == 0:
        return heading(ego_state)

    delta = x[:, :2].astype(float) - np.asarray(ego_state[:2], dtype=float)
    distance = np.linalg.norm(delta, axis=1)
    speed = x[:, 2].astype(float)
    angles = np.arctan2(x[:, 5], x[:, 6])

    mask = (
        (ids != track_id)
        & (distance > 0.5)
        & (distance <= max_distance)
        & (speed > 1.0)
    )

    if not np.any(mask):
        return heading(ego_state)

    local_angles = angles[mask]
    local_distance = distance[mask]
    local_speed = speed[mask]

    weights = np.exp(-local_distance / 10.0) * np.minimum(local_speed, 15.0)
    c = float(np.sum(weights * np.cos(local_angles)))
    s = float(np.sum(weights * np.sin(local_angles)))

    if math.hypot(c, s) < 1e-8:
        return heading(ego_state)

    return math.atan2(s, c)


def traffic_corridor_cost(candidate_paths, obs_states, flow_heading, ego_state):
    """
    Penalize leaving the local traffic corridor.

    Uses current positions of nearby recorded vehicles only. It does not use
    their future positions or the ego's recorded future trajectory.
    """
    candidate_count, horizon_steps, _ = candidate_paths.shape

    if len(obs_states) < 2:
        # If local traffic is sparse, use heading consistency rather than
        # inventing a corridor from insufficient evidence.
        displacement = candidate_paths[:, -1] - np.asarray(ego_state[:2], float)
        along = displacement[:, 0] * math.cos(flow_heading) + displacement[:, 1] * math.sin(flow_heading)
        across = -displacement[:, 0] * math.sin(flow_heading) + displacement[:, 1] * math.cos(flow_heading)
        return 0.015 * across ** 2 - 0.005 * along

    obs = np.asarray(obs_states, dtype=float)
    origin = np.asarray(ego_state[:2], dtype=float)
    c, s = math.cos(flow_heading), math.sin(flow_heading)

    # Coordinates in the local traffic-flow frame.
    rel = obs[:, :2] - origin[None, :]
    along = rel[:, 0] * c + rel[:, 1] * s
    across = -rel[:, 0] * s + rel[:, 1] * c

    # Only nearby moving vehicles contribute to the local corridor estimate.
    mask = (
        (np.abs(along) <= 22.0)
        & (np.abs(across) <= 12.0)
        & (obs[:, 2] > 0.5)
    )
    if np.count_nonzero(mask) < 2:
        displacement = candidate_paths[:, -1] - origin
        along_move = displacement[:, 0] * c + displacement[:, 1] * s
        across_move = -displacement[:, 0] * s + displacement[:, 1] * c
        return 0.02 * across_move ** 2 - 0.005 * along_move

    lane_offsets = across[mask]
    lane_weights = np.exp(-np.abs(along[mask]) / 15.0) * np.minimum(obs[mask, 2], 15.0)

    # Robust weighted center and spread of nearby traffic in the lateral axis.
    order = np.argsort(lane_offsets)
    sorted_offsets = lane_offsets[order]
    sorted_weights = lane_weights[order]
    cumulative = np.cumsum(sorted_weights)
    median_idx = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    center_offset = float(sorted_offsets[min(median_idx, len(sorted_offsets) - 1)])

    # Evaluate how far each point of each candidate lies from the traffic band.
    rel_path = candidate_paths - origin[None, None, :]
    path_along = rel_path[:, :, 0] * c + rel_path[:, :, 1] * s
    path_across = -rel_path[:, :, 0] * s + rel_path[:, :, 1] * c

    # Keep the ego close to the lateral traffic center, but allow modest
    # departures to avoid collisions.
    lateral_error = path_across - center_offset
    outside_band = np.maximum(0.0, np.abs(lateral_error) - 3.0)
    lateral_cost = np.mean(lateral_error ** 2 + 2.0 * outside_band ** 2, axis=1)

    # Discourage trajectories that turn backward relative to local traffic.
    backward = np.maximum(0.0, -path_along[:, -1])
    return 0.16 * lateral_cost + 1.5 * backward ** 2


def choose_action(
    state, pred, obs_states, obs_preds, dt, horizon, safety, flow_heading=None
):
    """
    Choose a bounded action using predicted collision risk and local traffic
    corridor alignment. The ego's future recorded position is never consulted.
    """
    speed = max(0.0, float(state[2]))
    ego_heading = heading(state)
    if flow_heading is None:
        flow_heading = ego_heading

    obs_states = np.asarray(obs_states, dtype=float).reshape(-1, 7)
    obs_preds = np.asarray(obs_preds, dtype=float).reshape(-1, 3)

    # Estimate a local traffic speed for stable forward motion.
    if len(obs_states):
        near = np.linalg.norm(obs_states[:, :2] - state[:2], axis=1) < 18.0
        local_speeds = obs_states[near, 2]
        traffic_speed = float(np.median(local_speeds)) if len(local_speeds) else speed
    else:
        traffic_speed = speed

    speed_target = float(np.clip(traffic_speed, 1.0, 15.0))
    speed_control = float(np.clip(0.8 * (speed_target - speed), -1.5, 1.5))
    nominal_lon = float(np.clip(0.30 * float(pred[1]) + 0.70 * speed_control, -2.0, 1.8))

    flow_error = wrap_angle(float(flow_heading) - ego_heading)
    desired_yaw_rate = float(np.clip(1.25 * flow_error, -0.20, 0.20))
    nominal_lat = float(np.clip(
        0.20 * float(pred[2]) + 0.80 * speed * desired_yaw_rate,
        -1.2, 1.2,
    ))

    longitudinal_offsets = (-1.0, -0.4, 0.0, 0.4, 0.8)
    lateral_offsets = (-0.45, -0.20, 0.0, 0.20, 0.45)
    actions = np.asarray([
        (
            np.clip(nominal_lon + dlon, -2.5, 2.0),
            np.clip(nominal_lat + dlat, -1.5, 1.5),
        )
        for dlon in longitudinal_offsets
        for dlat in lateral_offsets
    ], dtype=float)

    step = min(max(float(dt), 0.005), 0.04)
    sample_dt = min(float(dt), 0.08)
    times = np.arange(sample_dt, horizon + 1e-8, sample_dt)
    if len(times) == 0 or times[-1] < horizon - 1e-6:
        times = np.append(times, horizon)

    candidate_paths = rollout(state, actions, times, step)
    nominal_path = rollout(
        state, np.asarray([[nominal_lon, nominal_lat]], dtype=float), times, step
    )[0]
    future_obstacles = obstacle_rollouts(obs_states, obs_preds, times, step)

    if len(future_obstacles):
        distances = np.linalg.norm(
            candidate_paths[:, None, :, :] - future_obstacles[None, :, :, :],
            axis=-1,
        )
        min_distance = distances.min(axis=(1, 2))
        violation = (distances < safety).any(axis=(1, 2))
        penetration = np.maximum(0.0, safety - distances)
        max_penetration = penetration.max(axis=(1, 2))
        total_penetration = penetration.sum(axis=(1, 2))

        # Penalize the closest risky approach, not all vehicles in the scene.
        risk = np.maximum(0.0, safety + 1.0 - min_distance) ** 2
    else:
        min_distance = np.full(len(actions), np.inf)
        violation = np.zeros(len(actions), dtype=bool)
        max_penetration = np.zeros(len(actions))
        total_penetration = np.zeros(len(actions))
        risk = np.zeros(len(actions))

    deviation = candidate_paths - nominal_path[None, :, :]
    path_cost = np.mean(np.sum(deviation ** 2, axis=2), axis=1)
    action_cost = (
        0.20 * (actions[:, 0] - nominal_lon) ** 2
        + 0.45 * (actions[:, 1] - nominal_lat) ** 2
    )

    corridor_cost = traffic_corridor_cost(
        candidate_paths, obs_states, float(flow_heading), state
    )

    # Small forward-progress preference; corridor and collision terms dominate.
    displacement = candidate_paths[:, -1] - np.asarray(state[:2], dtype=float)
    progress = displacement[:, 0] * math.cos(flow_heading) + displacement[:, 1] * math.sin(flow_heading)
    progress_cost = -0.04 * progress

    score = (
        0.8 * path_cost
        + action_cost
        + 6.0 * risk
        + corridor_cost
        + progress_cost
    )

    safe_indices = np.flatnonzero(~violation)
    if len(safe_indices):
        best = int(safe_indices[np.argmin(score[safe_indices])])
        safe_available = True
    else:
        # No candidate satisfies the threshold: minimize penetration first.
        best = int(np.lexsort((score, total_penetration, max_penetration))[0])
        safe_available = False

    return (
        actions[best],
        bool(violation[best]),
        float(min_distance[best]),
        safe_available,
        times,
        future_obstacles,
        candidate_paths[best].copy(),
        nominal_path.copy(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=int, default=9)
    parser.add_argument("--start-frame", type=int, default=5000)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--track-id", type=int)
    parser.add_argument("--radius", type=float, default=80.0)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--interval-ms", type=int, default=40)
    parser.add_argument("--horizon", type=float, default=1.2)
    parser.add_argument("--safety-distance", type=float, default=3.5)
    parser.add_argument("--interaction-radius", type=float, default=40.0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--save-metrics", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.day <= 10:
        raise ValueError("--day must be 1..10")
    if min(args.dt, args.horizon, args.radius, args.safety_distance) <= 0:
        raise ValueError("dt, horizon, radius, and safety-distance must be positive")
    if args.frames < 1:
        raise ValueError("--frames must be >= 1")
    if args.interval_ms < 1:
        raise ValueError("--interval-ms must be >= 1")

    if not torch.cuda.is_available():
        torch.set_num_threads(args.torch_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    graph_path = GRAPH_DIR / f"20181101_d{args.day}_0800_0830.pt"
    for required in (graph_path, args.checkpoint):
        if not required.exists():
            raise FileNotFoundError(f"Required file not found: {required}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("PyTorch threads:", torch.get_num_threads())
    print("Graph bank:", graph_path)
    print("Checkpoint:", args.checkpoint)

    started = time.perf_counter()
    graphs = load_graphs(graph_path)
    print(f"Loaded {len(graphs):,} frames in {time.perf_counter() - started:.2f}s")
    model, mean, std = load_model(args.checkpoint, device)

    start = min(max(args.start_frame, 0), len(graphs) - 1)
    stop = min(len(graphs), start + args.frames)
    frame_ids = list(range(start, stop))
    first = graphs[start]
    first_ids = first["track_id"].numpy()
    first_x = first["x"].numpy()

    if args.track_id is None:
        moving = np.flatnonzero(first_x[:, 2] > 1.0)
        if len(moving):
            idx = int(moving[np.argmax(first_x[moving, 2])])
        else:
            idx = int(np.argmax(first_x[:, 2]))
        track_id = first_ids[idx].item()
    else:
        hit = np.flatnonzero(first_ids == args.track_id)
        if not len(hit):
            raise ValueError(f"Track ID {args.track_id} absent from start frame")
        idx = int(hit[0])
        track_id = first_ids[idx].item()

    # Initialize from the recorded track exactly once.
    sim_state = first["x"][idx].numpy().copy()
    sim_state = set_heading(sim_state, heading(sim_state))

    print(f"Frames: {len(frame_ids):,}; selected track ID: {track_id}")
    print("Mode: independent simulated ego + recorded surrounding traffic.")
    print("The ego is initialized once; future recorded ego positions are not controls.")

    records = []
    timings = []
    states = []
    paths = []
    nominal_rollouts = []
    chosen_rollouts = []
    obstacle_points = []
    simulated_path = []

    for k, frame in enumerate(frame_ids):
        frame_started = time.perf_counter()
        graph = graphs[frame]
        raw = graph["x"].numpy()
        raw_ids = graph["track_id"].numpy()
        ego_state = sim_state.copy()

        # The simulated ego is inserted into the GNN graph; recorded obstacles
        # are retained at their observed positions for this frame.
        sim_graph, ego_index = make_graph(
            graph, track_id, ego_state, args.interaction_radius
        )
        gnn_started = time.perf_counter()
        predictions = predict(model, sim_graph, mean, std, device)
        gnn_seconds = time.perf_counter() - gnn_started
        ego_prediction = predictions[ego_index]

        offsets = raw[:, :2] - ego_state[:2]
        visible = (
            (raw_ids != track_id)
            & (np.einsum("ij,ij->i", offsets, offsets) <= args.radius ** 2)
        )
        obstacle_indices = np.flatnonzero(visible)
        obstacle_states = raw[obstacle_indices].copy()

        simulated_ids = sim_graph["track_id"].numpy()
        lookup = {
            value.item() if hasattr(value, "item") else value: i
            for i, value in enumerate(simulated_ids)
        }
        obstacle_predictions = np.asarray([
            predictions[lookup[value.item() if hasattr(value, "item") else value]]
            for value in raw_ids[obstacle_indices]
        ], dtype=float).reshape(-1, 3)

        # Current-frame local flow only; no future position of the ego is used.
        flow = local_flow_heading(graph, ego_state, track_id)

        controller_started = time.perf_counter()
        (
            action, violation, min_distance, safe_available, times,
            future_obstacles, chosen_path, nominal_path,
        ) = choose_action(
            ego_state,
            ego_prediction,
            obstacle_states,
            obstacle_predictions,
            args.dt,
            args.horizon,
            args.safety_distance,
            flow_heading=flow,
        )
        controller_seconds = time.perf_counter() - controller_started

        nominal_next = integrate(
            ego_state, float(ego_prediction[1]), float(ego_prediction[2]), args.dt
        )
        chosen_next = integrate(ego_state, float(action[0]), float(action[1]), args.dt)

        # The only state update is the controller's chosen action.
        sim_state = chosen_next.copy()

        simulated_path.append(ego_state[:2].copy())
        simulated_path = simulated_path[-250:]
        states.append(ego_state.copy())
        paths.append(np.asarray(simulated_path).copy())
        nominal_rollouts.append(nominal_path.copy())
        chosen_rollouts.append(chosen_path.copy())

        recorded_hit = np.flatnonzero(raw_ids == track_id)
        if len(recorded_hit):
            recorded_xy = raw[int(recorded_hit[0]), :2].copy()
        else:
            recorded_xy = np.array([np.nan, np.nan])

        obstacle_points.append((
            raw[visible, :2].copy(),
            future_obstacles[:, -1, :].copy()
            if len(future_obstacles) else np.empty((0, 2)),
        ))

        frame_seconds = time.perf_counter() - frame_started
        records.append({
            "frame": frame,
            "track_id": track_id,
            "x": float(ego_state[0]),
            "y": float(ego_state[1]),
            "speed": float(ego_state[2]),
            "heading_rad": heading(ego_state),
            "gnn_next_speed": float(ego_prediction[0]),
            "gnn_lon_acc": float(ego_prediction[1]),
            "gnn_lat_acc": float(ego_prediction[2]),
            "gnn_nominal_x": float(nominal_next[0]),
            "gnn_nominal_y": float(nominal_next[1]),
            "chosen_next_x": float(chosen_next[0]),
            "chosen_next_y": float(chosen_next[1]),
            "controller_deviation_m": float(
                np.linalg.norm(chosen_next[:2] - nominal_next[:2])
            ),
            "nearby_obstacles": len(obstacle_states),
            "min_predicted_distance_m": min_distance,
            "candidate_safety_violation": violation,
            "safe_candidate_available": safe_available,
            "chosen_lon_acc": float(action[0]),
            "chosen_lat_acc": float(action[1]),
            "local_flow_heading_rad": float(flow),
            "heading_error_to_flow_rad": float(wrap_angle(flow - heading(ego_state))),
            "gnn_seconds": gnn_seconds,
            "controller_seconds": controller_seconds,
            "frame_compute_seconds": frame_seconds,
            "recorded_current_x": float(recorded_xy[0]),
            "recorded_current_y": float(recorded_xy[1]),
            "reference_guided": False,
            "independent_ego": True,
        })
        timings.append((frame_seconds, gnn_seconds, controller_seconds))

        if args.profile and (k == 0 or (k + 1) % 25 == 0):
            avg = np.mean(timings[-25:], axis=0)
            print(
                f"[profile] {k + 1}/{len(frame_ids)} frame={frame} "
                f"total={avg[0]:.4f}s GNN={avg[1]:.4f}s "
                f"controller={avg[2]:.4f}s obstacles={len(obstacle_states)}"
            )

    if not records:
        raise RuntimeError("No simulation frames were processed.")

    outdir = ROOT / "artifacts" / "human_driver_gnn_avoidance"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"d{args.day}_frame{start}_track{track_id}_independent.csv"

    if args.save_metrics:
        pd.DataFrame(records).to_csv(outfile, index=False)
        print(f"Saved {len(records):,} simulation records: {outfile}")

    avg = np.mean(timings, axis=0)
    print(
        f"Precomputation complete: {len(records):,} frames; "
        f"mean compute={avg[0]:.4f}s, GNN={avg[1]:.4f}s, "
        f"controller={avg[2]:.4f}s"
    )
    print(
        "Safety candidate fraction:",
        f"{np.mean([r['safe_candidate_available'] for r in records]):.1%}",
    )
    print(
        "Candidate violation fraction:",
        f"{np.mean([r['candidate_safety_violation'] for r in records]):.1%}",
    )

    # ---------------------------- Animation ----------------------------
    fig, ax = plt.subplots(figsize=(15, 9))
    fig.subplots_adjust(left=.06, right=.77, top=.91, bottom=.09)
    ax.set_title("Independent simulated ego in recorded pNEUMA traffic")
    ax.set_xlabel("x coordinate (m)")
    ax.set_ylabel("y coordinate (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=.18)

    traffic = ax.scatter([], [], s=24, c="gray", alpha=.72, edgecolors="none", zorder=2)
    ego = ax.scatter([], [], s=150, c="yellow", edgecolors="black", linewidths=1.2, zorder=10)
    nominal = ax.scatter([], [], s=100, c="orange", marker="D", edgecolors="black", zorder=9)
    chosen = ax.scatter([], [], s=180, c="cyan", marker="*", edgecolors="black", zorder=11)
    recorded_ego = ax.scatter([], [], s=85, c="limegreen", marker="x", linewidths=2, zorder=8)
    predicted = ax.scatter([], [], s=70, c="red", edgecolors="black", zorder=7)

    obstacle_lines = LineCollection([], colors="red", linewidths=1, alpha=.5, zorder=5)
    ax.add_collection(obstacle_lines)
    deviation_line, = ax.plot([], [], "--", color="magenta", lw=2)
    path_line, = ax.plot([], [], "-", color="goldenrod", lw=2)
    nominal_line, = ax.plot([], [], "--", color="darkorange", lw=2, alpha=.95)
    chosen_line, = ax.plot([], [], "-", color="deepskyblue", lw=2.5, alpha=.95)

    safety_circle = Circle(
        states[0][:2], args.safety_distance, fill=False,
        edgecolor="red", linestyle=":", alpha=.8
    )
    ax.add_patch(safety_circle)

    info = ax.text(
        1.02, .98, "", transform=ax.transAxes, va="top", ha="left",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=.5", facecolor="white", alpha=.94),
        clip_on=False,
    )

    legend_handles = [
        Line2D([], [], marker="o", linestyle="None", color="gray", label="Recorded surrounding vehicles"),
        Line2D([], [], marker="o", linestyle="None", color="yellow", markeredgecolor="black", label="Simulated ego"),
        Line2D([], [], marker="x", linestyle="None", color="limegreen", label="Recorded ego (reference only)"),
        Line2D([], [], marker="D", linestyle="None", color="orange", markeredgecolor="black", label="GNN nominal next position"),
        Line2D([], [], marker="*", linestyle="None", color="cyan", markeredgecolor="black", label="Controller next position"),
        Line2D([], [], linestyle="--", color="darkorange", label="GNN nominal rollout"),
        Line2D([], [], color="deepskyblue", label="Controller rollout"),
        Line2D([], [], color="goldenrod", label="Recent simulated ego path"),
        Line2D([], [], color="red", label="Approx. obstacle future"),
        Line2D([], [], linestyle="--", color="magenta", label="Nominal-to-chosen deviation"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, .62), fontsize=8)

    def update(i):
        row = records[i]
        frame = int(row["frame"])
        raw = graphs[frame]["x"].numpy()
        ids = graphs[frame]["track_id"].numpy()
        center = np.array([row["x"], row["y"]])

        visible = (
            (ids != track_id)
            & (np.linalg.norm(raw[:, :2] - center, axis=1) <= args.radius)
        )
        traffic.set_offsets(raw[visible, :2] if visible.any() else np.empty((0, 2)))
        ego.set_offsets(center.reshape(1, 2))

        if np.isfinite(row["recorded_current_x"]) and np.isfinite(row["recorded_current_y"]):
            recorded_ego.set_offsets([[
                row["recorded_current_x"], row["recorded_current_y"]
            ]])
        else:
            recorded_ego.set_offsets(np.empty((0, 2)))

        nominal.set_offsets([[row["gnn_nominal_x"], row["gnn_nominal_y"]]])
        chosen.set_offsets([[row["chosen_next_x"], row["chosen_next_y"]]])
        deviation_line.set_data(
            [row["gnn_nominal_x"], row["chosen_next_x"]],
            [row["gnn_nominal_y"], row["chosen_next_y"]],
        )
        path_line.set_data(paths[i][:, 0], paths[i][:, 1])
        nominal_line.set_data(nominal_rollouts[i][:, 0], nominal_rollouts[i][:, 1])
        chosen_line.set_data(chosen_rollouts[i][:, 0], chosen_rollouts[i][:, 1])
        safety_circle.center = center

        current_obstacles, future_obstacles = obstacle_points[i]
        predicted.set_offsets(
            future_obstacles if len(future_obstacles) else np.empty((0, 2))
        )
        obstacle_lines.set_segments([
            [current_obstacles[j], future_obstacles[j]]
            for j in range(min(len(current_obstacles), len(future_obstacles)))
        ] if len(future_obstacles) else [])

        ax.set_xlim(center[0] - args.radius, center[0] + args.radius)
        ax.set_ylim(center[1] - args.radius, center[1] + args.radius)

        row_number = i + 1
        info.set_text(
            f"D{args.day} | frame {frame} ({row_number}/{len(records)})\n"
            f"Track: {track_id}\n"
            f"Mode: independent ego\n"
            f"Speed: {row['speed']:.2f} m/s\n"
            f"GNN a_lon/a_lat: {row['gnn_lon_acc']:+.2f}/{row['gnn_lat_acc']:+.2f}\n"
            f"Chosen a_lon/a_lat: {row['chosen_lon_acc']:+.2f}/{row['chosen_lat_acc']:+.2f}\n"
            f"Nominal deviation: {row['controller_deviation_m']:.3f} m\n"
            f"Nearby obstacles: {row['nearby_obstacles']}\n"
            f"Min predicted distance: {row['min_predicted_distance_m']:.2f} m\n"
            f"Ego re-anchored: NO\n"
            f"Safe candidate: {'YES' if row['safe_candidate_available'] else 'NO'}\n"
            f"Chosen candidate violation: {'YES' if row['candidate_safety_violation'] else 'NO'}"
        )

        return (
            traffic, ego, nominal, chosen, recorded_ego, predicted,
            obstacle_lines, deviation_line, path_line, nominal_line,
            chosen_line, safety_circle, info,
        )

    animation = FuncAnimation(
        fig, update, frames=len(records), interval=args.interval_ms,
        blit=False, repeat=False, cache_frame_data=False,
    )
    # Keep a strong reference to the animation.
    fig._pneuma_animation = animation

    try:
        plt.show()
    finally:
        print(f"Replay complete: prepared {len(records)}/{len(frame_ids)} frames.")
        print("Recorded surrounding vehicles do not react to the simulated ego.")


if __name__ == "__main__":
    main()
