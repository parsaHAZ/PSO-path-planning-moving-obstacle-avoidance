from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

ROOT = Path(".")
GRAPH_DIR = ROOT / "data/pneuma/graphs_fixed_80ms"
CSV_DIR = ROOT / "data/pneuma/processed"
OUT_DIR = ROOT / "artifacts/pneuma_graph_label_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS = [1, 8, 9, 10]
GRAPHS_PER_DAY = 300
SEED = 20261023

# A source observation must be close to the graph's claimed timestamp.
# We report matching coverage rather than silently accepting distant rows.
TIME_TOLERANCE_S = 0.006

FEATURE_MAP = {
    "x_m": 0,
    "y_m": 1,
    "speed_mps": 2,
    "lon_acc": 3,
    "lat_acc": 4,
    "heading_sin": 5,
    "heading_cos": 6,
}
TARGET_MAP = {
    "speed_mps": 0,
    "lon_acc": 1,
    "lat_acc": 2,
}

def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def numpy_value(obj, key):
    value = obj[key]
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)

def nearest_index(times, query):
    """Index of nearest timestamp in a sorted one-dimensional array."""
    j = int(np.searchsorted(times, query))
    if j <= 0:
        return 0
    if j >= len(times):
        return len(times) - 1
    if abs(times[j - 1] - query) <= abs(times[j] - query):
        return j - 1
    return j

def summarize_errors(df, group_cols, value_col):
    rows = []
    for key, part in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        values = part[value_col].dropna().to_numpy(dtype=float)
        row = dict(zip(group_cols, key))
        row.update({
            "n": int(len(values)),
            "mae": float(np.mean(np.abs(values))) if len(values) else None,
            "rmse": float(np.sqrt(np.mean(values ** 2))) if len(values) else None,
            "p95_abs": float(np.quantile(np.abs(values), 0.95)) if len(values) else None,
            "max_abs": float(np.max(np.abs(values))) if len(values) else None,
        })
        rows.append(row)
    return rows

rng = np.random.default_rng(SEED)
all_records = []
day_summaries = []

for day in DAYS:
    graph_path = GRAPH_DIR / f"20181101_d{day}_0800_0830.pt"
    csv_path = CSV_DIR / f"20181101_d{day}_0800_0830.csv"

    print("\n" + "=" * 88)
    print(f"D{day}")
    print(f"Graph file: {graph_path} | exists={graph_path.exists()}")
    print(f"CSV file:   {csv_path} | exists={csv_path.exists()}")

    if not graph_path.exists() or not csv_path.exists():
        print("SKIPPED: required input file missing.")
        day_summaries.append({
            "day": day, "status": "missing_input",
            "graph_file_exists": graph_path.exists(),
            "csv_file_exists": csv_path.exists(),
        })
        continue

    source = pd.read_csv(csv_path, low_memory=False)
    required = {
        "track_id", "time", "x_m", "y_m", "speed_mps",
        "lon_acc", "lat_acc", "heading_sin", "heading_cos",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"D{day}: processed CSV missing columns: {missing}")

    source = source.replace([np.inf, -np.inf], np.nan)
    source["track_id"] = pd.to_numeric(source["track_id"], errors="coerce")
    source["time"] = pd.to_numeric(source["time"], errors="coerce")
    source = source.dropna(subset=["track_id", "time"])
    source["track_id"] = source["track_id"].astype(np.int64)
    source = source.sort_values(["track_id", "time"]).reset_index(drop=True)

    # Per-track arrays make lookup efficient without a large dataframe merge.
    tracks = {}
    for track_id, part in source.groupby("track_id", sort=False):
        part = part.sort_values("time")
        tracks[int(track_id)] = {
            "time": part["time"].to_numpy(dtype=np.float64),
            "values": part[
                ["x_m", "y_m", "speed_mps", "lon_acc", "lat_acc",
                 "heading_sin", "heading_cos"]
            ].to_numpy(dtype=np.float64),
        }

    graphs = load_pt(graph_path)
    if not isinstance(graphs, (list, tuple)) or not graphs:
        raise ValueError(f"D{day}: expected a non-empty list of graph records")

    n_pick = min(GRAPHS_PER_DAY, len(graphs))
    indices = np.unique(
        np.linspace(0, len(graphs) - 1, n_pick, dtype=int)
    )
    # Add a reproducible random subset to avoid checking only the timeline endpoints.
    random_indices = rng.choice(len(graphs), size=n_pick, replace=False)
    indices = np.unique(np.concatenate([indices, random_indices]))

    print(
        f"source_rows={len(source):,}, tracks={len(tracks):,}, "
        f"graphs={len(graphs):,}, sampled_graphs={len(indices):,}"
    )

    day_records = []
    missing_fields = set()

    for graph_idx in indices:
        graph = graphs[int(graph_idx)]
        required_graph_keys = ("time", "track_id", "x", "y", "target_dt")
        absent = [k for k in required_graph_keys if k not in graph]
        if absent:
            missing_fields.update(absent)
            continue

        graph_time = float(np.asarray(graph["time"]).reshape(-1)[0])
        ids = numpy_value(graph, "track_id").reshape(-1).astype(np.int64)
        x = numpy_value(graph, "x").astype(np.float64)
        y = numpy_value(graph, "y").astype(np.float64)
        dt = numpy_value(graph, "target_dt").reshape(-1).astype(np.float64)

        if len(ids) != len(x) or len(ids) != len(y) or len(ids) != len(dt):
            raise ValueError(
                f"D{day}, graph {graph_idx}: node count mismatch "
                f"ids={len(ids)} x={len(x)} y={len(y)} dt={len(dt)}"
            )

        for node_idx, track_id in enumerate(ids):
            base = {
                "day": day,
                "graph_idx": int(graph_idx),
                "graph_time": graph_time,
                "track_id": int(track_id),
                "node_idx": int(node_idx),
                "target_dt_s": float(dt[node_idx]),
            }

            track = tracks.get(int(track_id))
            if track is None or not len(track["time"]):
                day_records.append({
                    **base, "current_match": False, "future_match": False,
                    "current_time_error_s": np.nan,
                    "future_time_error_s": np.nan,
                })
                continue

            times = track["time"]
            values = track["values"]

            # Match the node to the source observation at its current graph time.
            ci = nearest_index(times, graph_time)
            current_error = abs(float(times[ci]) - graph_time)
            current_ok = current_error <= TIME_TOLERANCE_S

            rec = {
                **base,
                "source_current_time": float(times[ci]),
                "current_time_error_s": current_error,
                "current_match": bool(current_ok),
                "future_match": False,
                "future_time_error_s": np.nan,
            }

            if current_ok:
                for name, col in FEATURE_MAP.items():
                    actual = float(x[node_idx, col])
                    expected = float(values[ci, col])
                    rec[f"current_{name}_saved"] = actual
                    rec[f"current_{name}_source"] = expected
                    rec[f"current_{name}_abs_error"] = abs(actual - expected)

            # Independently locate the source observation at graph_time + target_dt.
            node_dt = float(dt[node_idx])
            if np.isfinite(node_dt) and node_dt > 0:
                target_time = graph_time + node_dt
                fi = nearest_index(times, target_time)
                future_error = abs(float(times[fi]) - target_time)
                future_ok = future_error <= TIME_TOLERANCE_S

                rec.update({
                    "claimed_target_time": target_time,
                    "source_future_time": float(times[fi]),
                    "future_time_error_s": future_error,
                    "future_match": bool(future_ok),
                })

                if future_ok:
                    for name, col in TARGET_MAP.items():
                        saved = float(y[node_idx, col])
                        expected = float(values[fi, FEATURE_MAP[name]])
                        rec[f"target_{name}_saved"] = saved
                        rec[f"target_{name}_source"] = expected
                        rec[f"target_{name}_abs_error"] = abs(saved - expected)

            day_records.append(rec)

    all_records.extend(day_records)

    day_df = pd.DataFrame(day_records)
    current_matches = (
        day_df["current_match"].fillna(False)
        if "current_match" in day_df else pd.Series([], dtype=bool)
    )
    future_matches = (
        day_df["future_match"].fillna(False)
        if "future_match" in day_df else pd.Series([], dtype=bool)
    )

    summary = {
        "day": day,
        "status": "ok",
        "source_rows": int(len(source)),
        "source_tracks": int(source["track_id"].nunique()),
        "graph_count": int(len(graphs)),
        "sampled_graph_count": int(len(indices)),
        "sampled_nodes": int(len(day_df)),
        "current_matches": int(current_matches.sum()),
        "current_match_rate": float(current_matches.mean()) if len(current_matches) else None,
        "future_matches": int(future_matches.sum()),
        "future_match_rate": float(future_matches.mean()) if len(future_matches) else None,
        "median_current_time_error_s": (
            float(day_df.loc[current_matches, "current_time_error_s"].median())
            if current_matches.any() else None
        ),
        "median_future_time_error_s": (
            float(day_df.loc[future_matches, "future_time_error_s"].median())
            if future_matches.any() else None
        ),
        "missing_graph_fields": sorted(missing_fields),
    }

    for name in FEATURE_MAP:
        col = f"current_{name}_abs_error"
        if col in day_df:
            vals = day_df.loc[current_matches, col].dropna()
            summary[f"current_{name}_mae"] = float(vals.mean()) if len(vals) else None

    for name in TARGET_MAP:
        col = f"target_{name}_abs_error"
        if col in day_df:
            vals = day_df.loc[future_matches, col].dropna()
            summary[f"target_{name}_mae"] = float(vals.mean()) if len(vals) else None
            summary[f"target_{name}_p95_abs_error"] = (
                float(vals.quantile(0.95)) if len(vals) else None
            )
            summary[f"target_{name}_max_abs_error"] = (
                float(vals.max()) if len(vals) else None
            )

    day_summaries.append(summary)
    print(json.dumps(summary, indent=2))

if not all_records:
    raise RuntimeError("No audit records produced. Check input files and graph schema.")

detail = pd.DataFrame(all_records)
detail_path = OUT_DIR / "sampled_node_comparisons.csv"
summary_path = OUT_DIR / "summary_by_day.json"
detail.to_csv(detail_path, index=False)
summary_path.write_text(
    json.dumps(day_summaries, indent=2, allow_nan=False),
    encoding="utf-8"
)

print("\n" + "=" * 88)
print("OVERALL LABEL AUDIT")
print("=" * 88)
print(f"Detailed records: {detail_path}")
print(f"Day summaries:    {summary_path}")

for day in DAYS:
    d = detail[detail["day"] == day]
    if d.empty:
        continue

    print(f"\nD{day}: sampled nodes={len(d):,}")
    for label, match_col, prefix in [
        ("current features", "current_match", "current_"),
        ("future targets", "future_match", "target_"),
    ]:
        matched = d[match_col].fillna(False)
        n_match = int(matched.sum())
        print(f"  {label}: source time matches={n_match:,}/{len(d):,}")
        if n_match == 0:
            continue

        for name in (
            list(FEATURE_MAP) if prefix == "current_" else list(TARGET_MAP)
        ):
            col = f"{prefix}{name}_abs_error"
            if col not in d:
                continue
            vals = d.loc[matched, col].dropna()
            if len(vals):
                print(
                    f"    {name:14s} MAE={vals.mean():.8g}, "
                    f"p95_abs={vals.quantile(.95):.8g}, "
                    f"max_abs={vals.max():.8g}, n={len(vals):,}"
                )

print("\nIMPORTANT:")
print("- This is a sampled audit, not yet an exhaustive all-node validation.")
print("- A source-time match means a same-track observation was found near the claimed time.")
print("- Large label errors despite close timestamp matches suggest label construction/alignment issues.")
print("- Low match rates may mean the saved target_dt does not correspond to source observation times.")
print("- No model or graph files were modified.")
