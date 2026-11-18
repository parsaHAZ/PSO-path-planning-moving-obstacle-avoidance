from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import torch

SCRIPT = Path("research/validate_and_train_pneuma_80ms.py")
OUT = Path("artifacts/human_driving_validated_80ms")
PRED_OUT = OUT / "heldout_paired_predictions.csv"
TRACK_OUT = OUT / "paired_track_metrics.csv"
BOOT_OUT = OUT / "paired_track_bootstrap.csv"
SUMMARY_OUT = OUT / "paired_model_comparison.csv"

BOOTSTRAPS = 2000
SEED = 20261023
CHUNK_ROWS = 200_000
FLUSH_GRAPHS = 500
TARGETS = ["speed", "lon_acc", "lat_acc"]
MODELS = ["persistence", "ego_mlp", "interaction_gnn"]

spec = importlib.util.spec_from_file_location("pneuma80", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

torch.set_num_threads(1)
device = mod.DEVICE
print("Device:", device, flush=True)

def load_checkpoint(path):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

norm = load_checkpoint(OUT / "normalization.pt")
stats = (
    norm["x_mean"], norm["x_std"], norm["y_mean"],
    norm["y_std"], norm["edge_mean"], norm["edge_std"],
)
ym = norm["y_mean"].to(device)
ys = norm["y_std"].to(device)

ego_ckpt = load_checkpoint(OUT / "best_ego_mlp.pt")
gnn_ckpt = load_checkpoint(OUT / "best_interaction_gnn.pt")

ego = mod.EgoMLP().to(device)
ego.load_state_dict(ego_ckpt["state_dict"])
ego.eval()

gnn = mod.InteractionGNN().to(device)
gnn.load_state_dict(gnn_ckpt["state_dict"])
gnn.eval()

print("Ego MLP selected epoch:", ego_ckpt.get("epoch"))
print("Interaction GNN selected epoch:", gnn_ckpt.get("epoch"))

columns = [
    "day", "graph_index", "graph_time", "track_id", "node_index"
]
for t in TARGETS:
    columns.extend([
        f"y_{t}", f"persistence_{t}", f"ego_mlp_{t}",
        f"interaction_gnn_{t}"
    ])

# Start a fresh output; never append to a partial file from a failed run.
if PRED_OUT.exists():
    PRED_OUT.unlink()

pending = []
pending_graphs = 0
total_rows = 0

def flush_predictions():
    global pending, pending_graphs, total_rows
    if not pending:
        return
    chunk = pd.concat(pending, ignore_index=True)
    numeric_cols = [
        c for c in columns
        if c not in ("day", "graph_index", "graph_time", "track_id", "node_index")
    ]
    values = chunk[numeric_cols].to_numpy(dtype=np.float64)
    keep = np.isfinite(values).all(axis=1)
    dropped = int((~keep).sum())
    if dropped:
        print("Dropping non-finite rows:", dropped, flush=True)
        chunk = chunk.loc[keep]
    chunk.to_csv(
        PRED_OUT, mode="a", index=False,
        header=(not PRED_OUT.exists() or PRED_OUT.stat().st_size == 0)
    )
    total_rows += len(chunk)
    pending = []
    pending_graphs = 0
    print(f"  saved {total_rows:,} prediction rows", flush=True)

with torch.no_grad():
    for day in (9, 10):
        graphs = mod.load_day(day)
        print(f"\nD{day}: loaded {len(graphs):,} graphs", flush=True)

        for gi, graph in enumerate(graphs):
            x, y, edge_index, edge_attr, valid = mod.unpack(graph)
            if x.shape[0] == 0 or not bool(valid.any()):
                continue

            X, Y, mask, ei, ea = mod.normalized_graph(
                [(x, y, edge_index, edge_attr, valid)], stats
            )

            pred_ego = ego(X, ei, ea) * ys + ym
            pred_gnn = gnn(X, ei, ea) * ys + ym
            target = Y * ys + ym
            persistence = x[:, [2, 3, 4]].to(device)

            ids = mod.tensor(graph, "track_id", torch.long)
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().numpy()
            else:
                ids = np.asarray(ids)

            times = graph.get("time", np.nan)
            if torch.is_tensor(times):
                times = times.detach().cpu()
                times = times.item() if times.numel() == 1 else np.nan
            times = float(times) if np.isscalar(times) else np.nan

            indices = torch.where(mask)[0].detach().cpu().numpy()
            if len(indices) == 0:
                continue

            target_np = target[mask].detach().cpu().numpy()
            p_np = persistence[mask].detach().cpu().numpy()
            ego_np = pred_ego[mask].detach().cpu().numpy()
            gnn_np = pred_gnn[mask].detach().cpu().numpy()

            data = {
                "day": np.full(len(indices), day, dtype=np.int16),
                "graph_index": np.full(len(indices), gi, dtype=np.int32),
                "graph_time": np.full(len(indices), times, dtype=np.float64),
                "track_id": ids[indices].astype(np.int64, copy=False),
                "node_index": indices.astype(np.int32, copy=False),
            }
            for k, t in enumerate(TARGETS):
                data[f"y_{t}"] = target_np[:, k]
                data[f"persistence_{t}"] = p_np[:, k]
                data[f"ego_mlp_{t}"] = ego_np[:, k]
                data[f"interaction_gnn_{t}"] = gnn_np[:, k]

            pending.append(pd.DataFrame(data, columns=columns))
            pending_graphs += 1

            if pending_graphs >= FLUSH_GRAPHS:
                flush_predictions()

            if (gi + 1) % 5000 == 0:
                print(f"  processed {gi + 1:,}/{len(graphs):,}", flush=True)

        flush_predictions()
        print(f"D{day} complete.", flush=True)

print("\nPrediction export complete:", PRED_OUT)
print("Total finite paired rows:", f"{total_rows:,}")
print("No model was retrained and no checkpoint was overwritten.")

if total_rows == 0:
    raise RuntimeError("No held-out predictions were exported.")

# Stream the prediction CSV. Aggregate sufficient statistics for
# node-weighted metrics and per-track mean absolute errors.
node_stats = {}
track_parts = []
read_rows = 0

print("\nReading prediction CSV in chunks for metrics...", flush=True)

for chunk in pd.read_csv(PRED_OUT, chunksize=CHUNK_ROWS):
    read_rows += len(chunk)

    for day in (9, 10):
        d = chunk[chunk["day"] == day]
        if d.empty:
            continue

        for t in TARGETS:
            y = d[f"y_{t}"].to_numpy(dtype=np.float64)

            for model in MODELS:
                e = d[f"{model}_{t}"].to_numpy(dtype=np.float64) - y
                key = (day, t, model)
                if key not in node_stats:
                    node_stats[key] = [0, 0.0, 0.0]
                node_stats[key][0] += len(e)
                node_stats[key][1] += float(np.abs(e).sum())
                node_stats[key][2] += float(np.square(e).sum())

                part = pd.DataFrame({
                    "day": d["day"].to_numpy(),
                    "track_id": d["track_id"].to_numpy(),
                    "target": t,
                    "model": model,
                    "ae": np.abs(e),
                })
                grouped = part.groupby(
                    ["day", "track_id", "target", "model"],
                    sort=False
                )["ae"].agg(["sum", "count"]).reset_index()
                track_parts.append(grouped)

    if read_rows % (CHUNK_ROWS * 5) < CHUNK_ROWS:
        print(f"  read {read_rows:,} rows", flush=True)

if read_rows != total_rows:
    print(f"WARNING: wrote {total_rows:,} rows but read {read_rows:,}")

# Combine chunk-level sums/counts so tracks crossing chunk boundaries
# receive their correct full-track MAE.
partial = pd.concat(track_parts, ignore_index=True)
track_metrics = (
    partial.groupby(
        ["day", "track_id", "target", "model"], as_index=False
    )[["sum", "count"]].sum()
)
track_metrics["track_mae"] = track_metrics["sum"] / track_metrics["count"]
track_metrics = track_metrics[
    ["day", "track_id", "target", "model", "count", "track_mae"]
]
track_metrics.to_csv(TRACK_OUT, index=False)

summary_rows = []
for (day, target, model), (n, abs_sum, sq_sum) in node_stats.items():
    summary_rows.append({
        "day": day,
        "target": target,
        "model": model,
        "n_rows": n,
        "mae": abs_sum / n,
        "rmse": np.sqrt(sq_sum / n),
    })

summary = pd.DataFrame(summary_rows)
track_counts = (
    track_metrics.groupby("day")["track_id"].nunique().to_dict()
)
summary["n_tracks"] = summary["day"].map(track_counts)
summary = summary[
    ["day", "target", "model", "n_rows", "n_tracks", "mae", "rmse"]
]
summary.to_csv(SUMMARY_OUT, index=False)

# Paired bootstrap: resample tracks within each day, comparing track MAE.
rng = np.random.default_rng(SEED)
bootstrap_rows = []

for day in (9, 10):
    day_tracks = track_metrics[track_metrics["day"] == day]

    for target in TARGETS:
        target_tracks = day_tracks[day_tracks["target"] == target]
        wide = target_tracks.pivot(
            index="track_id", columns="model", values="track_mae"
        ).dropna(subset=MODELS)

        n_tracks = len(wide)
        if n_tracks < 2:
            print(f"WARNING: insufficient tracks for D{day} {target}")
            continue

        for challenger, reference in [
            ("ego_mlp", "persistence"),
            ("interaction_gnn", "persistence"),
            ("interaction_gnn", "ego_mlp"),
        ]:
            delta = (
                wide[challenger].to_numpy()
                - wide[reference].to_numpy()
            )
            boot = np.empty(BOOTSTRAPS, dtype=np.float64)

            for b in range(BOOTSTRAPS):
                sample = rng.integers(0, n_tracks, size=n_tracks)
                boot[b] = delta[sample].mean()

            bootstrap_rows.append({
                "day": day,
                "target": target,
                "comparison": f"{challenger} - {reference}",
                "n_tracks": n_tracks,
                "mean_track_mae_difference": float(delta.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "bootstrap_replicates": BOOTSTRAPS,
            })

boot_df = pd.DataFrame(bootstrap_rows)
boot_df.to_csv(BOOT_OUT, index=False)

print("\n=== NODE-WEIGHTED HELD-OUT METRICS ===")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\n=== PAIRED TRACK-BOOTSTRAP RESULTS ===")
print("Difference = first model minus second; negative favors first model.")
print(boot_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nSaved outputs:")
for p in (PRED_OUT, TRACK_OUT, SUMMARY_OUT, BOOT_OUT):
    print(" ", p)
