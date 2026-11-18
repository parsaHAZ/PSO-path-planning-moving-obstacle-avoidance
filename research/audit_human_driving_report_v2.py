# research/audit_human_driving_report_v2.py
"""
Audit the v2 human-driving research report.

This script:
- Audits the v2 report, not v1.
- Checks the inherited report sections.
- Audits the legacy v1 bootstrap CSV separately.
- Checks whether the strict 80 ms evaluation protocol is documented.
- Checks whether the strict 80 ms metric and data-audit artifacts exist.
- Does not train models, load prediction CSVs, or generate plots.
"""

from pathlib import Path
import csv
import sys


REPORT = Path(
    "artifacts/human_driving_research_report_v2/"
    "human_driving_behavior_and_traffic_context_report.md"
)

# This is a legacy v1 analysis artifact, not evidence that v2 is correct.
LEGACY_BOOTSTRAP = Path(
    "artifacts/human_driving_condition_analysis_v1/"
    "paired_track_bootstrap_sparse0_5_vs_dense6plus.csv"
)

STRICT_METRICS = Path(
    "artifacts/human_driving_validated_80ms/metrics.csv"
)

STRICT_DATA_AUDIT = Path(
    "artifacts/human_driving_validated_80ms/data_audit.json"
)


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalized(text):
    return " ".join(text.lower().replace("-", " ").split())


errors = []
warnings = []


def fail(message):
    errors.append(message)
    print(f"FAIL: {message}")


def warn(message):
    warnings.append(message)
    print(f"WARN: {message}")


print("=== V2 REPORT ===")

if not REPORT.exists():
    raise SystemExit(f"FAIL: v2 report missing: {REPORT}")

report_text = REPORT.read_text(encoding="utf-8")
report_norm = normalized(report_text)

print(f"Report: {REPORT}")
print(f"Size: {REPORT.stat().st_size:,} bytes")
print(f"Lines: {len(report_text.splitlines()):,}")

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
    if section.lower() in report_text.lower():
        print(f"PASS: {section}")
    else:
        fail(f"Missing expected v2 report section: {section}")


print("\n=== STRICT 80 MS EVALUATION ARTIFACTS ===")

for path, label in [
    (STRICT_METRICS, "Strict 80 ms metrics"),
    (STRICT_DATA_AUDIT, "Strict 80 ms data audit"),
]:
    exists = path.exists()
    print(f"{'PASS' if exists else 'FAIL'}: {label}: {path}")
    if not exists:
        fail(f"{label} artifact missing: {path}")


print("\n=== STRICT 80 MS PROTOCOL DOCUMENTATION ===")

# These are checks for documentation, not proof of methodological validity.
protocol_checks = {
    "80 ms horizon": (
        "80 ms" in report_norm
        or "80ms" in report_norm
    ),
    "D1-D7 training split": (
        "d1" in report_norm
        and "d7" in report_norm
        and ("train" in report_norm or "training" in report_norm)
    ),
    "D8 validation split": (
        "d8" in report_norm
        and "valid" in report_norm
    ),
    "D9 held-out test split": (
        "d9" in report_norm
        and ("test" in report_norm or "held out" in report_norm
             or "heldout" in report_norm)
    ),
    "D10 exclusion is explained": (
        "d10" in report_norm
        and (
            "excluded" in report_norm
            or "83.417" in report_norm
            or "83.4 ms" in report_norm
        )
    ),
    "GNN and simpler baselines are discussed": (
        "interaction_gnn" in report_norm
        and "ego_mlp" in report_norm
        and "persistence" in report_norm
    ),
}

for label, ok in protocol_checks.items():
    print(f"{'PASS' if ok else 'WARN'}: {label}")
    if not ok:
        warn(f"Strict 80 ms protocol not clearly documented: {label}")


print("\n=== LEGACY V1 BOOTSTRAP AUDIT ===")

if not LEGACY_BOOTSTRAP.exists():
    warn(f"Legacy v1 bootstrap CSV missing: {LEGACY_BOOTSTRAP}")
else:
    rows = read_csv(LEGACY_BOOTSTRAP)
    print(f"Legacy bootstrap file: {LEGACY_BOOTSTRAP}")
    print(f"Rows: {len(rows)}")

    expected_direction = {
        "speed": -1,
        "next_speed_mps": -1,
        "lon_acc": -1,
        "next_lon_acc": -1,
        "lat_acc": 1,
        "next_lat_acc": 1,
    }

    if len(rows) != 18:
        fail(
            "Legacy v1 bootstrap expected 18 rows; "
            f"found {len(rows)}"
        )

    for row in rows:
        target = row.get("target", "")
        drone = row.get("drone", "?")
        horizon = row.get("horizon_ms", "?")

        try:
            contrast = float(row["contrast_sparse_minus_dense_pp"])
            low = float(row["contrast_ci_low"])
            high = float(row["contrast_ci_high"])
        except (KeyError, TypeError, ValueError) as exc:
            fail(
                f"Invalid legacy bootstrap row "
                f"{drone} {horizon} {target}: {exc}"
            )
            continue

        direction = expected_direction.get(target)

        if direction is None:
            fail(
                f"Unknown legacy bootstrap target: "
                f"{drone} {horizon} {target}"
            )
            continue

        sign_ok = (
            contrast < 0 if direction == -1 else contrast > 0
        )
        ci_ok = (
            high < 0 if direction == -1 else low > 0
        )

        print(
            f"{drone} {horizon}ms {target}: "
            f"contrast={contrast:+.3f} pp, "
            f"CI=[{low:+.3f}, {high:+.3f}], "
            f"direction={'PASS' if sign_ok else 'FAIL'}, "
            f"CI={'PASS' if ci_ok else 'FAIL'}"
        )

        if not sign_ok:
            fail(
                f"Unexpected legacy contrast direction: "
                f"{drone} {horizon} {target}"
            )

        if not ci_ok:
            fail(
                f"Legacy CI does not exclude zero in expected "
                f"direction: {drone} {horizon} {target}"
            )


print("\n=== FINAL RESULT ===")

if errors:
    print(f"FAIL: {len(errors)} error(s), {len(warnings)} warning(s)")
    for error in errors:
        print(f"ERROR: {error}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    sys.exit(1)

if warnings:
    print(
        f"PARTIAL PASS: report structure and available legacy checks "
        f"passed; {len(warnings)} documentation warning(s) remain."
    )
    print("Do not treat this as full validation of the research report.")
    sys.exit(2)

print("PASS: v2 report structure and documented protocol checks passed.")
print("Legacy bootstrap checks passed separately.")
print("This audit does not establish model validity or prove absence of leakage.")
