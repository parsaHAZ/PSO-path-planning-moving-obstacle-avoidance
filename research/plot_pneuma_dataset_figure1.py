
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "pneuma" / "processed"
OUTPUT_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAYS = range(1, 11)  # D1-D9; exclude D10
SEED = 20261031
MAX_FEATURE_ROWS = 500_000
MAX_TRAJECTORY_TRACKS = 100

rng = np.random.default_rng(SEED)

feature_columns = [
    "speed_mps",
    "lon_acc",
    "lat_acc",
    "heading_rad",
]

feature_samples = {column: [] for column in feature_columns}
trajectory_frames = []

for day in DAYS:
    path = DATA_DIR / f"20181101_d{day}_0800_0830.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing real pNEUMA data file: {path}")

    print(f"Reading D{day}: {path.name}")
    df = pd.read_csv(path)

    required = {
        "track_id", "x_m", "y_m",
        "speed_mps", "lon_acc", "lat_acc", "heading_rad"
    }
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing)}"
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    # Retain valid observations for the feature distributions.
    valid = df.dropna(subset=feature_columns)

    for column in feature_columns:
        values = valid[column].to_numpy(dtype=np.float64)

        # Deterministic sampling for manageable memory use.
        if len(values) > MAX_FEATURE_ROWS:
            indices = rng.choice(
                len(values), MAX_FEATURE_ROWS, replace=False
            )
            values = values[indices]

        feature_samples[column].append(values)

    # Retain trajectory coordinates and track identifiers.
    trajectory_frames.append(
        df[["track_id", "x_m", "y_m"]].dropna().copy()
    )

    print(
        f"  observations: {len(df):,}; "
        f"valid feature rows: {len(valid):,}"
    )

# Combine feature samples across D1-D9.
features = {
    column: np.concatenate(parts)
    for column, parts in feature_samples.items()
    if parts
}

# Combine trajectories and select complete tracks deterministically.
trajectories = pd.concat(trajectory_frames, ignore_index=True)

track_ids = trajectories["track_id"].drop_duplicates().to_numpy()

if len(track_ids) > MAX_TRAJECTORY_TRACKS:
    selected_ids = rng.choice(
        track_ids, MAX_TRAJECTORY_TRACKS, replace=False
    )
else:
    selected_ids = track_ids

trajectory_subset = trajectories[
    trajectories["track_id"].isin(selected_ids)
]

# Convert heading to degrees for an interpretable distribution.
heading_degrees = np.degrees(features["heading_rad"])
heading_degrees = (heading_degrees + 180) % 360 - 180

# Publication-style multi-panel figure.
fig, axes = plt.subplots(
    2, 3, figsize=(16, 10), constrained_layout=True
)

# Panel (a): recorded trajectories.
ax = axes[0, 0]

for _, track in trajectory_subset.groupby("track_id", sort=False):
    track = track.dropna(subset=["x_m", "y_m"])
    if len(track) >= 2:
        ax.plot(
            track["x_m"],
            track["y_m"],
            linewidth=0.45,
            alpha=0.55
        )

ax.set_title("(a) Recorded vehicle trajectories")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(alpha=0.2)

# Panel (b): speed.
ax = axes[0, 1]
ax.hist(features["speed_mps"], bins=80, density=True)
ax.set_title("(b) Speed distribution")
ax.set_xlabel("Speed (m/s)")
ax.set_ylabel("Density")
ax.grid(alpha=0.2)

# Panel (c): longitudinal acceleration.
ax = axes[0, 2]
ax.hist(features["lon_acc"], bins=80, density=True)
ax.set_title("(c) Longitudinal acceleration")
ax.set_xlabel("Longitudinal acceleration (m/s²)")
ax.set_ylabel("Density")
ax.grid(alpha=0.2)

# Panel (d): lateral acceleration.
ax = axes[1, 0]
ax.hist(features["lat_acc"], bins=80, density=True)
ax.set_title("(d) Lateral acceleration")
ax.set_xlabel("Lateral acceleration (m/s²)")
ax.set_ylabel("Density")
ax.grid(alpha=0.2)

# Panel (e): heading.
ax = axes[1, 1]
ax.hist(heading_degrees, bins=72, density=True)
ax.set_title("(e) Heading distribution")
ax.set_xlabel("Heading (degrees)")
ax.set_ylabel("Density")
ax.grid(alpha=0.2)

# Panel (f): dataset coverage.
ax = axes[1, 2]
ax.axis("off")

coverage_text = (
    "pNEUMA recorded observations\n\n"
    "Recording date: 1 November 2018\n"
    "Recording period: 08:00–08:30\n"
    "Sections: D1–D9\n"
    "Training: D1–D7\n"
    "Validation: D8\n"
    "Held-out test: D9\n"
    "D10: excluded from strict 80 ms benchmark\n\n"
    f"Observations: {len(trajectories):,}\n"
    f"Displayed trajectories: {len(selected_ids):,}\n"
    "Feature distributions: sampled observations"
)

ax.text(
    0.05, 0.95, coverage_text,
    va="top", ha="left",
    fontsize=11,
    transform=ax.transAxes
)

png_path = OUTPUT_DIR / "figure1_pneuma_dataset.png"
# pdf_path = OUTPUT_DIR / "figure1_pneuma_dataset.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
# fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print("\nFigure generation completed.")
print(f"PNG: {png_path}")
# print(f"PDF: {pdf_path}")
print(f"Trajectory tracks displayed: {len(selected_ids):,}")

for column, values in features.items():
    print(
        f"{column}: n={len(values):,}, "
        f"mean={np.mean(values):.5f}, "
        f"std={np.std(values):.5f}"
    )
