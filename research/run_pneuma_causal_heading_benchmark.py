from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path.cwd()
PY = sys.executable

RAW = ROOT / "data/pneuma/raw"
PROC = ROOT / "data/pneuma/processed_causal_heading"
GRAPHS = ROOT / "data/pneuma/graphs_all_vehicles_80ms_causal_heading"
OUT = ROOT / "artifacts/human_driving_validated_80ms_causal_heading"
RESEARCH = ROOT / "research"

PREPROCESS = RESEARCH / "pneuma_preprocess.py"
BUILDER_SRC = RESEARCH / "build_pneuma_all_vehicle_graphs_80ms.py"
TRAIN_SRC = RESEARCH / "validate_and_train_pneuma_80ms.py"
EXPORT_SRC = RESEARCH / "export_pneuma_paired_predictions.py"

BUILDER = RESEARCH / "build_pneuma_all_vehicle_graphs_80ms_causal_heading.py"
TRAIN = RESEARCH / "validate_and_train_pneuma_80ms_causal_heading.py"
EXPORT = RESEARCH / "export_pneuma_paired_predictions_causal_heading.py"

def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")

def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}; "
            "no modified script will be executed."
        )
    return source.replace(old, new, 1)

for p in (PREPROCESS, BUILDER_SRC, TRAIN_SRC, EXPORT_SRC):
    require_file(p)

for d in (PROC, GRAPHS, OUT):
    if d.exists() and any(d.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite existing corrected-run directory: {d}"
        )
    d.mkdir(parents=True, exist_ok=True)

# 1. Regenerate processed data with the corrected causal-heading function.
raw_files = sorted(
    RAW.glob("20181101_d*_0800_0830.csv"),
    key=lambda p: int(p.stem.split("_d")[1].split("_")[0])
)
if len(raw_files) != 10:
    raise RuntimeError(
        f"Expected 10 raw pNEUMA day files; found {len(raw_files)} in {RAW}"
    )

print("\n========== 1/5: PREPROCESS RAW DAYS ==========", flush=True)
for i, raw in enumerate(raw_files, 1):
    out_csv = PROC / raw.name
    print(f"\n[{i}/10] {raw.name}", flush=True)
    subprocess.run(
        [
            PY, str(PREPROCESS),
            "--input", str(raw),
            "--output", str(out_csv),
        ],
        cwd=ROOT,
        check=True,
    )

# 2. Make a separate graph builder that reads corrected CSVs and writes
#    only to the dedicated corrected graph directory.
builder_text = BUILDER_SRC.read_text(encoding="utf-8")
builder_text = replace_once(
    builder_text,
    'SOURCE_DIR = ROOT / "data" / "pneuma" / "processed"',
    f'SOURCE_DIR = ROOT / "data" / "pneuma" / "processed_causal_heading"',
    "builder source directory",
)
builder_text = replace_once(
    builder_text,
    'OUTPUT_DIR = ROOT / "data" / "pneuma" / "graphs_all_vehicles_80ms"',
    f'OUTPUT_DIR = ROOT / "data" / "pneuma" / "graphs_all_vehicles_80ms_causal_heading"',
    "builder output directory",
)
BUILDER.write_text(builder_text, encoding="utf-8", newline="\n")

print("\n========== 2/5: BUILD CORRECTED GRAPHS ==========", flush=True)
subprocess.run([PY, str(BUILDER)], cwd=ROOT, check=True)

# Ensure all expected graph files exist before training.
expected_graphs = [
    GRAPHS / f"20181101_d{day}_0800_0830.pt"
    for day in range(1, 11)
]
missing = [str(p) for p in expected_graphs if not p.is_file()]
if missing:
    raise RuntimeError("Graph build incomplete; missing:\n" + "\n".join(missing))

# 3. Train in a new output directory, reading the corrected all-vehicle graphs.
train_text = TRAIN_SRC.read_text(encoding="utf-8")
train_text = replace_once(
    train_text,
    'GRAPH_DIR = ROOT / "data" / "pneuma" / "graphs_fixed_80ms"',
    'GRAPH_DIR = ROOT / "data" / "pneuma" / "graphs_all_vehicles_80ms_causal_heading"',
    "trainer graph directory",
)
train_text = replace_once(
    train_text,
    'OUT = ROOT / "artifacts" / "human_driving_validated_80ms"',
    'OUT = ROOT / "artifacts" / "human_driving_validated_80ms_causal_heading"',
    "trainer output directory",
)
TRAIN.write_text(train_text, encoding="utf-8", newline="\n")

print("\n========== 3/5: TRAIN AND EVALUATE ==========", flush=True)
subprocess.run([PY, str(TRAIN)], cwd=ROOT, check=True)

for name in ("metrics.csv", "normalization.pt",
             "best_ego_mlp.pt", "best_interaction_gnn.pt"):
    p = OUT / name
    if not p.is_file():
        raise RuntimeError(f"Training did not produce required artifact: {p}")

# 4. Export paired predictions and recompute per-track bootstrap statistics.
export_text = EXPORT_SRC.read_text(encoding="utf-8")
export_text = replace_once(
    export_text,
    'SCRIPT = Path("research/validate_and_train_pneuma_80ms.py")',
    'SCRIPT = Path("research/validate_and_train_pneuma_80ms_causal_heading.py")',
    "exporter trainer import",
)
export_text = replace_once(
    export_text,
    'OUT = Path("artifacts/human_driving_validated_80ms")',
    'OUT = Path("artifacts/human_driving_validated_80ms_causal_heading")',
    "exporter output directory",
)
EXPORT.write_text(export_text, encoding="utf-8", newline="\n")

print("\n========== 4/5: EXPORT PAIRED METRICS ==========", flush=True)
subprocess.run([PY, str(EXPORT)], cwd=ROOT, check=True)

for name in (
    "heldout_paired_predictions.csv",
    "paired_track_metrics.csv",
    "paired_track_bootstrap.csv",
):
    p = OUT / name
    if not p.is_file():
        raise RuntimeError(f"Export did not produce required artifact: {p}")

# 5. Report the actual output locations and the row-weighted metrics.
import pandas as pd

metrics = pd.read_csv(OUT / "metrics.csv")
print("\n========== 5/5: CORRECTED RUN SUMMARY ==========")
print("Processed CSVs:", PROC)
print("Graphs:", GRAPHS)
print("Artifacts:", OUT)
print("\nMetrics:")
print(metrics.to_string(index=False))

print("\nBootstrap:")
boot = pd.read_csv(OUT / "paired_track_bootstrap.csv")
print(boot.to_string(index=False))

print("\nCORRECTED PIPELINE COMPLETED.")
print("These results use causal-heading preprocessing.")
print("The existing paper report has NOT been overwritten.")
