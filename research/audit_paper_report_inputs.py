from pathlib import Path
import csv

root = Path.cwd()
report = root / "reports" / "pneuma_human_driver_prediction_results.md"

if not report.exists():
    raise FileNotFoundError(f"Report not found: {report}")

text = report.read_text(encoding="utf-8")

text = text.replace("D1?D7", "D1–D7")
text = text.replace("D9?D10", "D9–D10")
text = text.replace(
    "- Report downstream planning results separately from human-motion\n"
    "\n"
    "---",
    "- Report downstream planning results separately from human-motion\n"
    "  prediction results; do not infer planning benefits from prediction\n"
    "  metrics alone.\n"
    "\n"
    "---",
)

report.write_text(text, encoding="utf-8")

print("========== REPORT CHECK ==========")
print(f"Updated: {report}")
print(f"Size: {report.stat().st_size:,} bytes")
for token in ("D1?D7", "D9?D10"):
    print(f"Contains {token!r}: {token in text}")

print("\n========== EXISTING EXPERIMENT FILES ==========")
roots = [
    root / "artifacts",
    root / "reports",
    root / "research",
]

keywords = (
    "heldout_predictions",
    "predictions.csv",
    "metrics",
    "summary",
    "evaluation",
    "audit",
    "human_driving_adaptation",
    "human_driving_horizon",
    "human_driving_regime",
    "human_driving_interpretation",
)

found = []
for base in roots:
    if not base.exists():
        continue
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if any(k in name for k in keywords):
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            found.append((str(p.relative_to(root)), size))

for name, size in sorted(found):
    print(f"{size:>12,} bytes  {name}")

print("\n========== PREDICTION CSV SCHEMAS ==========")
csv_files = [
    root / "artifacts" / "human_driving_adaptation_v2" /
    "heldout_predictions.csv",
    root / "artifacts" / "human_driving_interpretation_v1" /
    "heldout_predictions.csv",
]

for p in csv_files:
    if not p.exists():
        print(f"MISSING: {p.relative_to(root)}")
        continue
    try:
        with p.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        print(f"\nFILE: {p.relative_to(root)}")
        print("COLUMNS:", ", ".join(header))
        print(f"SIZE: {p.stat().st_size:,} bytes")
    except Exception as exc:
        print(f"ERROR reading {p}: {exc}")

print("\nInventory complete. No experiments were rerun.")
