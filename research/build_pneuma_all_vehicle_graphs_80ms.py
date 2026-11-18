from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree


ROOT = Path.cwd()
SOURCE_DIR = ROOT / "data" / "pneuma" / "processed"
OUTPUT_DIR = ROOT / "data" / "pneuma" / "graphs_all_vehicles_80ms"

DAYS = range(1, 11)
HORIZON_S = 0.080
TOLERANCE_S = 0.0041
RADIUS_M = 40.0
MIN_NODES = 2

FEATURES = [
    "x_m",
    "y_m",
    "speed_mps",
    "lon_acc",
    "lat_acc",
    "heading_sin",
    "heading_cos",
]
TARGETS = ["speed_mps", "lon_acc", "lat_acc"]
REQUIRED = ["track_id", "time"] + FEATURES

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def numeric_clean(df):
    for col in set(REQUIRED + TARGETS):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def make_targets(df):
    """
    For every source row, find the nearest strictly later same-track
    observation to current_time + 80 ms. Accept it only when its actual
    time difference from the requested horizon is <= 4.1 ms.

    Targets are future speed, longitudinal acceleration, and lateral
    acceleration. Rows without an accepted future observation receive
    NaN targets and NaN target_dt.
    """
    n = len(df)
    y = np.full((n, 3), np.nan, dtype=np.float32)
    dt_out = np.full(n, np.nan, dtype=np.float32)

    usable = (
        df["track_id"].notna().to_numpy()
        & df["time"].notna().to_numpy()
    )
    row_ids = np.flatnonzero(usable)

    if len(row_ids) == 0:
        return y, dt_out

    # Target lookup may use a future row even if its current-position
    # features are incomplete; only the target fields must be finite.
    work = df.iloc[row_ids][["track_id", "time"] + TARGETS].copy()
    work["_row_id"] = row_ids
    work = work.sort_values(
        ["track_id", "time", "_row_id"], kind="mergesort"
    )

    for _, group in work.groupby("track_id", sort=False):
        times = group["time"].to_numpy(dtype=np.float64)
        row_index = group["_row_id"].to_numpy(dtype=np.int64)
        target_values = group[TARGETS].to_numpy(dtype=np.float64)

        finite_target = np.isfinite(target_values).all(axis=1)
        if finite_target.sum() < 1:
            continue

        # Search only among observations whose target values are finite.
        candidate_times = times[finite_target]
        candidate_rows = row_index[finite_target]
        candidate_values = target_values[finite_target]

        order = np.argsort(candidate_times, kind="mergesort")
        candidate_times = candidate_times[order]
        candidate_rows = candidate_rows[order]
        candidate_values = candidate_values[order]

        desired = times + HORIZON_S
        right = np.searchsorted(candidate_times, desired, side="left")

        # Compare the first candidate at/after the requested time with
        # the preceding candidate, choosing the closest valid future row.
        for local_i, original_row in enumerate(row_index):
            pos = int(right[local_i])
            current_time = times[local_i]
            best = -1
            best_error = np.inf

            for candidate_pos in (pos - 1, pos):
                if candidate_pos < 0 or candidate_pos >= len(candidate_times):
                    continue

                candidate_time = candidate_times[candidate_pos]
                actual_dt = candidate_time - current_time

                if actual_dt <= 0:
                    continue

                error = abs(actual_dt - HORIZON_S)
                if error < best_error:
                    best_error = error
                    best = candidate_pos

            if best < 0 or best_error > TOLERANCE_S:
                continue

            y[original_row] = candidate_values[best].astype(np.float32)
            dt_out[original_row] = np.float32(
                candidate_times[best] - current_time
            )

    return y, dt_out


def make_edges(x):
    """
    Build directed edges for all pairs within the 40 m radius.
    Edge attributes are destination minus source:
      [relative_x, relative_y, distance, relative_speed]
    """
    n = len(x)
    if n < 2:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
        )

    positions = x[:, :2].astype(np.float64)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(RADIUS_M, output_type="ndarray")

    if pairs.size == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
        )

    i = pairs[:, 0]
    j = pairs[:, 1]
    delta = positions[j] - positions[i]
    distance = np.sqrt(np.sum(delta * delta, axis=1))

    keep = (distance > 0.0) & (distance <= RADIUS_M)
    i, j = i[keep], j[keep]
    delta, distance = delta[keep], distance[keep]

    if len(i) == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
        )

    relative_speed = x[j, 2] - x[i, 2]

    # Direction i -> j: destination j minus source i.
    src_forward = i
    dst_forward = j
    attr_forward = np.column_stack(
        [delta[:, 0], delta[:, 1], distance, relative_speed]
    )

    # Direction j -> i: destination i minus source j.
    src_reverse = j
    dst_reverse = i
    attr_reverse = np.column_stack(
        [-delta[:, 0], -delta[:, 1], distance, -relative_speed]
    )

    edge_index = np.vstack([
        np.concatenate([src_forward, src_reverse]),
        np.concatenate([dst_forward, dst_reverse]),
    ])
    edge_attr = np.vstack([attr_forward, attr_reverse])

    return (
        torch.as_tensor(edge_index, dtype=torch.long),
        torch.as_tensor(edge_attr, dtype=torch.float32),
    )


def main():
    overall_start = time.time()
    summary = {
        "builder": "build_pneuma_all_vehicle_graphs_80ms.py",
        "source_dir": str(SOURCE_DIR),
        "output_dir": str(OUTPUT_DIR),
        "horizon_s": HORIZON_S,
        "tolerance_s": TOLERANCE_S,
        "radius_m": RADIUS_M,
        "node_cap": None,
        "features": FEATURES,
        "targets": TARGETS,
        "days": {},
    }

    for day in DAYS:
        started = time.time()
        source_path = SOURCE_DIR / (
            f"20181101_d{day}_0800_0830.csv"
        )
        output_path = OUTPUT_DIR / (
            f"20181101_d{day}_0800_0830.pt"
        )

        if not source_path.exists():
            raise FileNotFoundError(source_path)

        print(f"\nD{day}: reading {source_path}", flush=True)
        df = pd.read_csv(source_path)
        missing = sorted(set(REQUIRED) - set(df.columns))
        if missing:
            raise ValueError(f"D{day}: missing required columns: {missing}")

        df = numeric_clean(df)
        df["_row_id"] = np.arange(len(df), dtype=np.int64)

        print(f"D{day}: generating future labels", flush=True)
        all_y, all_dt = make_targets(df)
        df["_target_speed"] = all_y[:, 0]
        df["_target_lon_acc"] = all_y[:, 1]
        df["_target_lat_acc"] = all_y[:, 2]
        df["_target_dt"] = all_dt

        # Match the original current-node rule: all seven current node
        # features, track_id, and time must be finite.
        finite_current = np.isfinite(
            df[["track_id", "time"] + FEATURES].to_numpy(dtype=np.float64)
        ).all(axis=1)
        current = df.loc[finite_current].copy()

        current["track_id"] = current["track_id"].astype(np.int64)
        current = current.sort_values(
            ["time", "track_id"], kind="mergesort"
        )

        graphs = []
        total_nodes = 0
        total_edges = 0
        frames_over_80 = 0
        matched_targets = 0
        max_nodes = 0

        print(
            f"D{day}: valid current observations={len(current):,}; "
            "building uncapped graphs",
            flush=True,
        )

        for frame_time, frame in current.groupby("time", sort=True):
            if len(frame) < MIN_NODES:
                continue

            x = frame[FEATURES].to_numpy(dtype=np.float32)
            track_ids = frame["track_id"].to_numpy(dtype=np.int64)
            y = frame[
                ["_target_speed", "_target_lon_acc", "_target_lat_acc"]
            ].to_numpy(dtype=np.float32)
            target_dt = frame["_target_dt"].to_numpy(dtype=np.float32)

            edge_index, edge_attr = make_edges(x)

            graph = {
                "time": torch.tensor(float(frame_time), dtype=torch.float64),
                "drone_id": torch.tensor(day, dtype=torch.long),
                "track_id": torch.from_numpy(track_ids.copy()),
                "x": torch.from_numpy(x.copy()),
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "y": torch.from_numpy(y.copy()),
                "target_dt": torch.from_numpy(target_dt.copy()),
            }
            graphs.append(graph)

            n = len(frame)
            total_nodes += n
            total_edges += edge_index.shape[1]
            max_nodes = max(max_nodes, n)
            frames_over_80 += int(n > 80)
            matched_targets += int(np.isfinite(target_dt).sum())

        # Do not overwrite anything in graphs_fixed_80ms.
        torch.save(graphs, output_path)

        record = {
            "source_rows": int(len(df)),
            "valid_current_vehicle_observations": int(len(current)),
            "graph_count": int(len(graphs)),
            "saved_nodes": int(total_nodes),
            "saved_directed_edges": int(total_edges),
            "frames_over_80_nodes": int(frames_over_80),
            "max_nodes_in_graph": int(max_nodes),
            "nodes_with_matched_targets": int(matched_targets),
            "output_file": str(output_path),
            "seconds": round(time.time() - started, 2),
        }
        summary["days"][f"D{day}"] = record

        print(json.dumps({f"D{day}": record}, indent=2), flush=True)

        del df, current, graphs

    summary["total_seconds"] = round(time.time() - overall_start, 2)
    summary_path = OUTPUT_DIR / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========== BUILD COMPLETE ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
