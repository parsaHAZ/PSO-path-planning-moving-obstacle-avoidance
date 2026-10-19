from pathlib import Path
import csv
import sys

report = Path(
    "artifacts/human_driving_research_report_v1/"
    "human_driving_history_traffic_context_report.md"
)
bootstrap_path = Path(
    "artifacts/human_driving_condition_analysis_v1/"
    "paired_track_bootstrap_sparse0_5_vs_dense6plus.csv"
)

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

if not report.exists():
    raise SystemExit(f"FAIL: report missing: {report}")
if not bootstrap_path.exists():
    raise SystemExit(f"FAIL: bootstrap CSV missing: {bootstrap_path}")

text = report.read_text(encoding="utf-8")
rows = read_csv(bootstrap_path)
errors = []

print("=== REPORT FILE ===")
print(f"Report exists: {report.exists()}")
print(f"Report size: {report.stat().st_size:,} bytes")
print(f"Bootstrap rows: {len(rows)}")

required_sections = [
    "## Abstract",
    "## Data and protocol",
    "## History prediction metrics",
    "## Shuffled-history ablation",
    "## Traffic-context evaluation",
    "## Track-bootstrap results",
    "## Robustness: sparse degree 0-5 versus dense degree 6+",
    "## Limitations",
    "## Conclusions",
]
for section in required_sections:
    ok = section in text
    print(f"{'PASS' if ok else 'FAIL'}: {section}")
    if not ok:
        errors.append(f"Missing section: {section}")

print("\n=== BOOTSTRAP DIRECTION AND CI AUDIT ===")
expected = {
    "speed": -1,
    "next_speed_mps": -1,
    "lon_acc": -1,
    "next_lon_acc": -1,
    "lat_acc": 1,
    "next_lat_acc": 1,
}
for row in rows:
    target = row.get("target", "")
    drone = row.get("drone", "?")
    horizon = row.get("horizon_ms", "?")

    try:
        contrast = float(row["contrast_sparse_minus_dense_pp"])
        low = float(row["contrast_ci_low"])
        high = float(row["contrast_ci_high"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"Invalid numeric row: {row}; {exc}")
        print(f"FAIL: {drone} {horizon} {target}: invalid numbers")
        continue

    direction = expected.get(target)
    sign_ok = (
        direction == -1 and contrast < 0
    ) or (
        direction == 1 and contrast > 0
    )
    ci_ok = (
        high < 0 if direction == -1 else
        low > 0 if direction == 1 else False
    )

    print(
        f"{drone} {horizon}ms {target}: "
        f"contrast={contrast:+.3f} pp, "
        f"CI=[{low:+.3f}, {high:+.3f}], "
        f"sign={'PASS' if sign_ok else 'FAIL'}, "
        f"CI={'PASS' if ci_ok else 'FAIL'}"
    )

    if not sign_ok:
        errors.append(f"Unexpected contrast direction: {drone} {horizon} {target}")
    if not ci_ok:
        errors.append(f"CI does not exclude zero in expected direction: {drone} {horizon} {target}")

if len(rows) != 18:
    errors.append(f"Expected 18 bootstrap rows; found {len(rows)}")

print("\n=== FINAL RESULT ===")
if errors:
    print(f"FAIL: {len(errors)} issue(s)")
    for error in errors:
        print("-", error)
    sys.exit(1)

print("PASS: all 18 bootstrap contrasts have the expected direction")
print("PASS: all 18 confidence intervals exclude zero in the expected direction")
print("PASS: required report sections exist")
print("No model retraining, prediction CSV loading, or plots.")
