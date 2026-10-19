from pathlib import Path
import csv
import re

report = Path(
    "artifacts/human_driving_research_report_v1/"
    "human_driving_history_traffic_context_report.md"
)

sources = {
    "history_metrics": Path(
        "artifacts/human_driving_history_v1/all_horizons_metrics.csv"
    ),
    "ablation_robustness": Path(
        "artifacts/human_driving_history_ablation_v2/robustness_summary.csv"
    ),
    "condition_metrics": Path(
        "artifacts/human_driving_condition_analysis_v1/condition_metrics_200ms.csv"
    ),
    "broad_bootstrap": Path(
        "artifacts/human_driving_condition_analysis_v1/"
        "paired_track_bootstrap_sparse0_5_vs_dense6plus.csv"
    ),
}

def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

print("=== SOURCE FILE CHECK ===")
source_rows = {}
for name, path in sources.items():
    exists = path.exists()
    rows = read_rows(path) if exists else []
    source_rows[name] = rows
    print(f"{name}: exists={exists}, rows={len(rows)}, path={path}")

print("\n=== REQUIRED REPORT CONTENT ===")
if not report.exists():
    raise SystemExit(f"REPORT MISSING: {report}")

text = report.read_text(encoding="utf-8")
checks = {
    "history metrics table": "all_horizons_metrics.csv" in text,
    "shuffled-history ablation": "Shuffled-history ablation" in text,
    "traffic-context section": "Traffic-context evaluation" in text,
    "broad bootstrap table": "contrast_sparse_minus_dense_pp" not in text
        and "paired_track_bootstrap_sparse0_5_vs_dense6plus.csv" in text,
    "limitations": "## Limitations" in text,
    "causal caveat": "does not establish" in text.lower()
        or "not causal" in text.lower(),
}
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}: {name}")

print("\n=== NUMERIC SOURCE CHECKS ===")
for name, rows in source_rows.items():
    if not rows:
        print(f"FAIL: {name} has no rows")
        continue

    empty_cells = sum(
        1 for row in rows for value in row.values()
        if value is None or not str(value).strip()
    )
    print(f"{name}: {len(rows)} rows; blank cells={empty_cells}")

history = source_rows["history_metrics"]
if history:
    models = sorted({r.get("model", "") for r in history})
    horizons = sorted({r.get("horizon_ms", "") for r in history})
    drones = sorted({r.get("drone", "") for r in history})
    print("Models:", models)
    print("Horizons:", horizons)
    print("Drones:", drones)

bootstrap = source_rows["broad_bootstrap"]
if bootstrap:
    print("\nBroad bootstrap CI check:")
    for row in bootstrap:
        try:
            lo = float(row["contrast_ci_low"])
            hi = float(row["contrast_ci_high"])
            contrast = float(row["contrast_sparse_minus_dense_pp"])
        except (KeyError, ValueError, TypeError) as exc:
            print("FAIL: invalid bootstrap row:", exc, row)
            continue

        target = row.get("target", "?")
        drone = row.get("drone", "?")
        horizon = row.get("horizon_ms", "?")
        expected_sign = -1 if target == "speed" or target == "next_speed_mps" else (
            -1 if target in {"lon_acc", "next_lon_acc"} else
            1 if target in {"lat_acc", "next_lat_acc"} else 0
        )
        ci_excludes_zero = lo > 0 or hi < 0
        sign_matches = expected_sign == 0 or (
            contrast < 0 if expected_sign < 0 else contrast > 0
        )
        print(
            f"{drone} {horizon} {target}: contrast={contrast:.3f}, "
            f"CI=[{lo:.3f}, {hi:.3f}], excludes_zero={ci_excludes_zero}, "
            f"expected_sign_matches={sign_matches}"
        )

print("\n=== CONCLUSION LANGUAGE CHECK ===")
for phrase in (
    "always improves",
    "proves that surrounding vehicles cause",
    "learned human preferences from pNEUMA",
    "inferred driver intention",
):
    print(f"{'WARN' if phrase.lower() in text.lower() else 'PASS'}: "
          f"phrase not present: {phrase}")

print("\nAudit finished. No models retrained; no prediction CSVs loaded; no plots created.")
