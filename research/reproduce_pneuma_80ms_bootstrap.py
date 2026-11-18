from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd

ROOT = Path("artifacts/human_driving_validated_80ms")
PRED = ROOT / "heldout_paired_predictions.csv"
SAVED_TRACK = ROOT / "paired_track_metrics.csv"
SAVED_BOOT = ROOT / "paired_track_bootstrap.csv"

SEED = 20261023
N_BOOT = 2000
CHUNK = 200_000
TARGETS = ["speed", "lon_acc", "lat_acc"]
MODELS = ["persistence", "ego_mlp", "interaction_gnn"]
COMPARISONS = [
    ("ego_mlp", "persistence"),
    ("interaction_gnn", "persistence"),
    ("interaction_gnn", "ego_mlp"),
]

if not all(p.exists() for p in [PRED, SAVED_TRACK, SAVED_BOOT]):
    raise FileNotFoundError("One or more required benchmark artifacts are missing.")

key_cols = ["day", "graph_index", "track_id", "node_index"]
numeric_cols = [
    f"y_{t}" for t in TARGETS
] + [
    f"{m}_{t}" for t in TARGETS for m in MODELS
]
usecols = key_cols + numeric_cols

# Track-level absolute-error sums and counts, accumulated across chunks.
sums = {}
counts = {}
day_rows = {}
nonfinite_rows = 0

# SQLite primary key detects duplicate prediction identities across chunks.
db_file = tempfile.NamedTemporaryFile(
    prefix="pneuma_prediction_key_audit_", suffix=".sqlite", delete=False
).name
conn = sqlite3.connect(db_file)
conn.execute("PRAGMA journal_mode=OFF")
conn.execute("PRAGMA synchronous=OFF")
conn.execute("""
    CREATE TABLE prediction_keys (
        day INTEGER,
        graph_index INTEGER,
        track_id INTEGER,
        node_index INTEGER,
        PRIMARY KEY(day, graph_index, track_id, node_index)
    )
""")

try:
    for chunk_no, df in enumerate(
        pd.read_csv(PRED, usecols=usecols, chunksize=CHUNK), 1
    ):
        for day, n in df.groupby("day", sort=False).size().items():
            day_rows[int(day)] = day_rows.get(int(day), 0) + int(n)

        keys = df[key_cols].to_numpy()
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO prediction_keys VALUES (?, ?, ?, ?)",
            (tuple(map(int, row)) for row in keys),
        )
        inserted = conn.total_changes - before
        duplicates = len(df) - inserted
        if duplicates:
            print(f"WARNING: found {duplicates:,} duplicate keys in chunk {chunk_no}")

        y = df[[f"y_{t}" for t in TARGETS]].to_numpy(dtype=np.float64)
        preds = {
            m: df[[f"{m}_{t}" for t in TARGETS]].to_numpy(dtype=np.float64)
            for m in MODELS
        }

        common = np.isfinite(y).all(axis=1)
        for arr in preds.values():
            common &= np.isfinite(arr).all(axis=1)
        nonfinite_rows += int((~common).sum())

        if common.any():
            valid = df.loc[common, ["day", "track_id"]]
            for model, arr in preds.items():
                err = np.abs(y[common] - arr[common])
                for j, target in enumerate(TARGETS):
                    tmp = pd.DataFrame({
                        "day": valid["day"].to_numpy(),
                        "track_id": valid["track_id"].to_numpy(),
                        "abs_error": err[:, j],
                    })
                    grouped = tmp.groupby(
                        ["day", "track_id"], sort=False
                    )["abs_error"].agg(["sum", "count"])

                    key = (model, target)
                    sums.setdefault(key, {})
                    counts.setdefault(key, {})
                    for (day, track_id), row in grouped.iterrows():
                        tk = (int(day), int(track_id))
                        sums[key][tk] = sums[key].get(tk, 0.0) + float(row["sum"])
                        counts[key][tk] = counts[key].get(tk, 0) + int(row["count"])

        if chunk_no % 5 == 0:
            conn.commit()
            print(f"Processed approximately {chunk_no * CHUNK:,} rows")

    conn.commit()
    duplicate_count = sum(day_rows.values()) - conn.execute(
        "SELECT COUNT(*) FROM prediction_keys"
    ).fetchone()[0]
finally:
    conn.close()
    try:
        Path(db_file).unlink()
    except OSError:
        pass

print("\n=== INPUT INTEGRITY ===")
for day in sorted(day_rows):
    print(f"D{day}: {day_rows[day]:,} prediction rows")
print(f"Rows with any non-finite target/prediction: {nonfinite_rows:,}")
print(f"Duplicate prediction keys: {duplicate_count:,}")

# Build the independently recomputed per-track MAE table.
track_rows = []
for (model, target), vals in sums.items():
    for (day, track_id), abs_sum in vals.items():
        n = counts[(model, target)][(day, track_id)]
        track_rows.append({
            "day": day,
            "track_id": track_id,
            "target": target,
            "model": model,
            "count": n,
            "track_mae": abs_sum / n,
        })
fresh = pd.DataFrame(track_rows)

saved_track = pd.read_csv(SAVED_TRACK)
join_keys = ["day", "track_id", "target", "model"]
merged = saved_track.merge(
    fresh,
    on=join_keys,
    how="outer",
    suffixes=("_saved", "_fresh"),
    indicator=True,
)
print("\n=== PER-TRACK METRIC RECONCILIATION ===")
print(merged["_merge"].value_counts().to_string())
both = merged[merged["_merge"] == "both"]
for col in ["count", "track_mae"]:
    a = both[f"{col}_saved"].to_numpy(dtype=float)
    b = both[f"{col}_fresh"].to_numpy(dtype=float)
    print(
        f"{col}: max absolute difference = "
        f"{np.max(np.abs(a-b)) if len(a) else float('nan'):.12g}"
    )

# Reproduce the original bootstrap's loop order and RNG sequence.
rng = np.random.default_rng(SEED)
recomputed_boot = []

for day in (9, 10):
    for target in TARGETS:
        subset = fresh[
            (fresh["day"] == day) & (fresh["target"] == target)
        ]
        wide = subset.pivot(
            index="track_id", columns="model", values="track_mae"
        ).dropna(subset=MODELS)

        n_tracks = len(wide)
        for challenger, reference in COMPARISONS:
            delta = (
                wide[challenger].to_numpy(dtype=np.float64)
                - wide[reference].to_numpy(dtype=np.float64)
            )
            boot = np.empty(N_BOOT, dtype=np.float64)
            for b in range(N_BOOT):
                sample = rng.integers(0, n_tracks, size=n_tracks)
                boot[b] = delta[sample].mean()

            recomputed_boot.append({
                "day": day,
                "target": target,
                "comparison": f"{challenger} - {reference}",
                "n_tracks": n_tracks,
                "mean_track_mae_difference": float(delta.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "bootstrap_replicates": N_BOOT,
            })

fresh_boot = pd.DataFrame(recomputed_boot)
saved_boot = pd.read_csv(SAVED_BOOT)
keys = ["day", "target", "comparison"]
comparison = saved_boot.merge(
    fresh_boot,
    on=keys,
    how="outer",
    suffixes=("_saved", "_fresh"),
    indicator=True,
)

print("\n=== BOOTSTRAP REPRODUCIBILITY ===")
print(comparison["_merge"].value_counts().to_string())
both = comparison[comparison["_merge"] == "both"]

for col in [
    "n_tracks",
    "mean_track_mae_difference",
    "ci95_low",
    "ci95_high",
    "bootstrap_replicates",
]:
    a = both[f"{col}_saved"].to_numpy(dtype=float)
    b = both[f"{col}_fresh"].to_numpy(dtype=float)
    delta = np.abs(a - b)
    print(
        f"{col}: max absolute difference = "
        f"{delta.max() if len(delta) else float('nan'):.12g}"
    )

print("\nRecomputed D9 GNN-minus-MLP results:")
print(
    fresh_boot[
        (fresh_boot["day"] == 9)
        & (fresh_boot["comparison"] == "interaction_gnn - ego_mlp")
    ].to_string(index=False, float_format=lambda x: f"{x:.9f}")
)

# Explicit pass/fail summary.
pass_checks = (
    duplicate_count == 0
    and nonfinite_rows == 0
    and len(merged) == len(saved_track)
    and (merged["_merge"] == "both").all()
    and len(comparison) == len(saved_boot)
    and (comparison["_merge"] == "both").all()
)
for col in ["n_tracks", "bootstrap_replicates"]:
    a = both[f"{col}_saved"].to_numpy(dtype=float)
    b = both[f"{col}_fresh"].to_numpy(dtype=float)
    pass_checks = pass_checks and np.array_equal(a, b)
for col in ["mean_track_mae_difference", "ci95_low", "ci95_high"]:
    a = both[f"{col}_saved"].to_numpy(dtype=float)
    b = both[f"{col}_fresh"].to_numpy(dtype=float)
    pass_checks = pass_checks and np.allclose(a, b, rtol=0, atol=1e-12)

print("\nOVERALL:", "PASS" if pass_checks else "CHECK REQUIRED")
if not pass_checks:
    raise SystemExit(2)
