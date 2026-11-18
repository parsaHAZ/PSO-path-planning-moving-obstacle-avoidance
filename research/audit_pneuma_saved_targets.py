from pathlib import Path
import numpy as np
import pandas as pd
import torch

root = Path("data/pneuma/graphs_fixed_80ms")
days = [1, 8, 9, 10]

def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def arr(g, key):
    v = g[key]
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    return np.asarray(v)

for day in days:
    path = root / f"20181101_d{day}_0800_0830.pt"
    graphs = load(path)
    print("\n" + "=" * 88)
    print(f"D{day}: {path}")
    print(f"graph_count={len(graphs)}")

    if not graphs:
        print("EMPTY")
        continue

    g0 = graphs[0]
    print("first_graph_keys:", list(g0.keys()))
    for key in ("time", "track_id", "x", "y", "target_dt"):
        if key not in g0:
            print(f"{key}: MISSING")
            continue
        a = arr(g0, key)
        print(
            f"{key}: shape={a.shape}, dtype={a.dtype}, "
            f"sample={a.reshape(-1)[:8].tolist()}"
        )

    # Check whether each target can be reproduced by looking up the
    # same track in a graph at current_time + target_dt.
    times = []
    for g in graphs:
        if "time" not in g:
            break
        t = arr(g, "time").reshape(-1)
        if len(t) == 0:
            break
        times.append(float(t[0]))

    if len(times) != len(graphs):
        print("Cannot perform future-graph matching: missing/empty time field.")
        continue

    times = np.asarray(times, dtype=np.float64)
    diffs = np.diff(times)
    print(
        "graph_time_delta:",
        f"min={np.min(diffs):.9g}, median={np.median(diffs):.9g}, "
        f"max={np.max(diffs):.9g}"
    )

    # Map timestamps to graph indices. This tests recorded observations;
    # it does not assume interpolated targets.
    time_to_idx = {}
    for i, t in enumerate(times):
        time_to_idx.setdefault(float(t), i)

    rng = np.random.default_rng(2026)
    candidate_graphs = np.linspace(
        0, len(graphs) - 1, min(250, len(graphs)), dtype=int
    )
    rng.shuffle(candidate_graphs)

    abs_errors = []
    matched = 0
    attempted = 0
    dt_values = []
    time_unit_note = "Check printed graph_time_delta and time samples."

    for i in candidate_graphs:
        g = graphs[i]
        required = ("track_id", "x", "y", "target_dt", "time")
        if any(k not in g for k in required):
            continue

        ids = arr(g, "track_id").reshape(-1)
        x = arr(g, "x")
        y = arr(g, "y")
        dt = arr(g, "target_dt").reshape(-1)
        current_t = times[i]

        # Group by target offset so each target-time lookup is shared.
        valid = (
            np.isfinite(dt)
            & np.isfinite(y).all(axis=1)
            & np.isfinite(x).all(axis=1)
        )
        for offset in np.unique(np.round(dt[valid], 6)):
            node_idx = np.flatnonzero(valid & (np.abs(dt - offset) < 5e-7))
            if len(node_idx) == 0:
                continue

            target_t = current_t + float(offset)
            j = int(np.argmin(np.abs(times - target_t)))
            time_error = abs(times[j] - target_t)

            # Only compare to a recorded graph close to the claimed target
            # time. Tolerance is half the median graph sampling interval.
            tol = max(float(np.median(np.abs(diffs))) * 0.51, 1e-6)
            if time_error > tol:
                continue

            future = graphs[j]
            future_ids = arr(future, "track_id").reshape(-1)
            future_x = arr(future, "x")
            future_lookup = {
                int(track_id): k for k, track_id in enumerate(future_ids)
            }

            for k in node_idx:
                track_id = int(ids[k])
                q = future_lookup.get(track_id)
                if q is None:
                    continue
                # Current target columns are assumed to represent future
                # speed, longitudinal acceleration and lateral acceleration.
                attempted += 1
                err = np.abs(y[k, :3] - future_x[q, [2, 3, 4]])
                if np.isfinite(err).all():
                    abs_errors.append(err)
                    matched += 1
                    dt_values.append(float(dt[k]))

    print(f"future_observation_pairs_matched={matched:,}")
    print(f"pairs_attempted={attempted:,}")
    if abs_errors:
        e = np.asarray(abs_errors)
        print("MAE | saved y - same-track future x:")
        print(f"  speed: {e[:, 0].mean():.8f}")
        print(f"  lon_acc: {e[:, 1].mean():.8f}")
        print(f"  lat_acc: {e[:, 2].mean():.8f}")
        print(
            "matched target_dt range:",
            f"{min(dt_values):.9f} to {max(dt_values):.9f} seconds"
        )
    else:
        print("No usable matches. Do not interpret this as a label failure yet;")
        print("the graph time units or target interpolation may differ.")

print("\nDiagnostic complete. No files were modified.")
