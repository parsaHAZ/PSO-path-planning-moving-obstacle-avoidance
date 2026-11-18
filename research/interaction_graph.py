from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


NODE_FEATURES = [
    "x_m",
    "y_m",
    "speed_mps",
    "lon_acc",
    "lat_acc",
    "heading_sin",
    "heading_cos",
]

TARGET_FEATURES = [
    "next_speed_mps",
    "next_lon_acc",
    "next_lat_acc",
]


def add_temporal_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["track_id", "time"]).copy()

    out["next_speed_mps"] = out.groupby("track_id")["speed_mps"].shift(-1)
    out["next_lon_acc"] = out.groupby("track_id")["lon_acc"].shift(-1)
    out["next_lat_acc"] = out.groupby("track_id")["lat_acc"].shift(-1)
    out["next_dt"] = out.groupby("track_id")["time"].shift(-1) - out["time"]

    return out


def ensure_heading_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "heading_rad" not in out.columns:
        dx = out.groupby("track_id")["x_m"].diff()
        dy = out.groupby("track_id")["y_m"].diff()
        out["heading_rad"] = np.arctan2(dy, dx)

    out["heading_rad"] = out.groupby("track_id")["heading_rad"].transform(
        lambda s: s.interpolate(limit_direction="both").fillna(0.0)
    )

    out["heading_sin"] = np.sin(out["heading_rad"])
    out["heading_cos"] = np.cos(out["heading_rad"])

    return out


def build_graphs(df: pd.DataFrame, radius_m: float = 40.0, max_nodes: int = 80, min_nodes: int = 2):
    df = ensure_heading_components(df)
    df = add_temporal_targets(df)

    required_columns = NODE_FEATURES + TARGET_FEATURES + ["next_dt"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_columns)

    graphs = []

    # One graph per timestamp
    for timestamp, frame in df.groupby("time", sort=True):
        if len(frame) < min_nodes:
            continue

        frame = frame.sort_values("track_id").head(max_nodes).reset_index(drop=True)

        xy = frame[["x_m", "y_m"]].to_numpy(dtype=np.float32)
        speed = frame["speed_mps"].to_numpy(dtype=np.float32)

        # Pairwise relative positions
        delta = xy[None, :, :] - xy[:, None, :]
        distances = np.linalg.norm(delta, axis=2)

        # Directed edges: vehicle i -> vehicle j
        src, dst = np.where((distances <= radius_m) & (distances > 0.0))
        if len(src) == 0:
            continue

        relative_position = xy[dst] - xy[src]
        relative_speed = speed[dst] - speed[src]

        edge_attr = np.column_stack([
            relative_position[:, 0],
            relative_position[:, 1],
            distances[src, dst],
            relative_speed,
        ]).astype(np.float32)

        node_features = frame[NODE_FEATURES].to_numpy(dtype=np.float32)
        targets = frame[TARGET_FEATURES].to_numpy(dtype=np.float32)

        graph = {
            "time": float(timestamp),
            "track_id": frame["track_id"].to_numpy(dtype=np.int64),
            "x": torch.from_numpy(node_features),
            "edge_index": torch.from_numpy(
                np.vstack([src, dst]).astype(np.int64)
            ),
            "edge_attr": torch.from_numpy(edge_attr),
            "y": torch.from_numpy(targets),
        }

        graphs.append(graph)

    return graphs


def save_graphs(graphs, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--radius", type=float, default=40.0)
    parser.add_argument("--max-nodes", type=int, default=80)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    graphs = build_graphs(df, radius_m=args.radius, max_nodes=args.max_nodes)
    save_graphs(graphs, args.output)

    total_nodes = sum(graph["x"].shape[0] for graph in graphs)
    total_edges = sum(graph["edge_index"].shape[1] for graph in graphs)

    print(f"Graphs: {len(graphs):,}")
    print(f"Nodes: {total_nodes:,}")
    print(f"Edges: {total_edges:,}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()