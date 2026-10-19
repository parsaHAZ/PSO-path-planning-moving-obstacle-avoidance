from pathlib import Path
import csv
import json
from datetime import datetime

ROOT = Path(".")
OUT = Path("artifacts/human_driving_research_report_v1")
OUT.mkdir(parents=True, exist_ok=True)

CONDITION_DIR = Path("artifacts/human_driving_condition_analysis_v1")
HISTORY_DIR = Path("artifacts/human_driving_history_v1")
ABLATION_DIR = Path("artifacts/human_driving_history_ablation_v2")

def read_csv(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def show_csv(path, max_rows=40):
    rows = read_csv(path)
    if rows is None:
        return f"Not found: `{path.as_posix()}`\n"
    if not rows:
        return f"`{path.as_posix()}` exists but contains no rows.\n"
    cols = list(rows[0].keys())
    lines = [
        f"Source: `{path.as_posix()}`",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows[:max_rows]:
        vals = [str(row.get(c, "")).replace("|", "\\|").replace("\n", " ") for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(rows)} rows.")
    return "\n".join(lines) + "\n"

def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

def section_from_candidates(title, candidates, max_rows=40):
    p = first_existing(candidates)
    if p is None:
        paths = "\n".join(f"- `{Path(x).as_posix()}`" for x in candidates)
        return f"## {title}\n\nNo matching output CSV found. Expected one of:\n{paths}\n"
    return f"## {title}\n\n{show_csv(p, max_rows)}"

def find_summary_files(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in {".json", ".txt", ".md"}
    )

def summarize_jsons(folder):
    files = find_summary_files(folder)
    if not files:
        return f"No JSON/TXT/MD summary files found in `{Path(folder).as_posix()}`.\n"
    out = []
    for p in files[:12]:
        out.append(f"### `{p.as_posix()}`")
        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text(encoding="utf-8-sig"))
                out.append("```json")
                out.append(json.dumps(obj, indent=2, ensure_ascii=False)[:7000])
                out.append("```")
            except Exception as e:
                out.append(f"Could not parse JSON: {e}")
        else:
            out.append(p.read_text(encoding="utf-8-sig", errors="replace")[:5000])
        out.append("")
    return "\n".join(out)

lines = []
lines.append("# Human-Driven Trajectory Prediction: History and Traffic-Context Evaluation")
lines.append("")
lines.append(f"Report generated: {datetime.now().astimezone().isoformat(timespec='seconds')}")
lines.append("")
lines.append("## Abstract")
lines.append("")
lines.append(
    "This report consolidates existing experiments on predicting human-driven vehicle "
    "motion from pNEUMA trajectories. It evaluates temporal history features, a shuffled-history "
    "control, and whether the relative predictive performance of ego-history and interaction-history "
    "models varies with graph-based traffic context. The evaluation uses D1–D7 for training, D8 "
    "for validation, and D9–D10 as held-out drones. This is predictive analysis: results do not "
    "establish causal effects of surrounding vehicles, driver intent, or human preferences."
)
lines.append("")
lines.append("## Data and evaluation protocol")
lines.append("")
lines.append("- Dataset: pNEUMA, 2018-11-01, 08:00–08:30.")
lines.append("- Split: D1–D7 train, D8 validation, D9–D10 held-out test.")
lines.append("- Prediction horizons: 200 ms, 500 ms, and 1000 ms.")
lines.append("- Targets: speed, longitudinal acceleration, and lateral acceleration.")
lines.append("- Traffic context: graph degree from a 40 m graph-construction radius.")
lines.append("- Sparse context in the robustness analysis: degree 0–5; dense context: degree 6+.")
lines.append("- Uncertainty estimates: track-cluster bootstrap; tracks, rather than individual rows, are resampled.")
lines.append("")
lines.append("## Research questions")
lines.append("")
lines.append("1. Does recent motion history improve prediction over ego-only baselines?")
lines.append("2. Does a shuffled-history control weaken performance, as expected if temporal alignment matters?")
lines.append("3. Does the relative error of the interaction-history model vary between sparse and dense graph contexts?")
lines.append("")
lines.append("## History and model comparisons")
lines.append("")
lines.append(
    "The history experiments use observations available at or before the prediction time. "
    "History features include past motion measurements and current-minus-past differences. "
    "The interaction-history model includes ego history plus graph-neighbor information; "
    "comparisons with ego-history are predictive comparisons and are not causal attribution."
)
lines.append("")
lines.append(section_from_candidates(
    "History evaluation summaries",
    [
        HISTORY_DIR / "summary.csv",
        HISTORY_DIR / "history_metrics.csv",
        HISTORY_DIR / "metrics.csv",
        HISTORY_DIR / "summary.json",
        HISTORY_DIR / "results.csv",
    ],
))
lines.append("")
lines.append(section_from_candidates(
    "Shuffled-history control",
    [
        Path("artifacts/human_driving_history_ablation_v2/shuffled_history_metrics.csv"),
        Path("artifacts/human_driving_history_ablation_v2/ablation_metrics.csv"),
        Path("artifacts/human_driving_history_ablation_v2/summary.csv"),
        Path("artifacts/human_driving_history_ablation_v2/results.csv"),
    ],
))
lines.append("")
lines.append("### Existing shuffled-history findings")
lines.append("")
lines.append(
    "The existing ablation results show that shuffling history substantially degrades performance, "
    "especially at longer horizons. This supports the importance of temporal alignment in the "
    "history inputs. Shuffled-history errors can be very large, so they should be interpreted as a "
    "control rather than as a realistic deployment baseline."
)
lines.append("")
lines.append("## Traffic-context evaluation")
lines.append("")
lines.append(
    "Graph degree is used as an operational proxy for local graph connectivity. It is not a "
    "universal measure of traffic density, and it depends on the graph radius and graph-building "
    "procedure. The main comparison is the percentage change in MAE for interaction-history relative "
    "to ego-history. Negative values favor interaction-history; positive values favor ego-history."
)
lines.append("")
lines.append(section_from_candidates(
    "Traffic-context metrics",
    [
        CONDITION_DIR / "condition_metrics_200ms.csv",
        CONDITION_DIR / "condition_metrics.csv",
        CONDITION_DIR / "traffic_condition_metrics.csv",
    ],
    max_rows=80,
))
lines.append("")
lines.append("## Bootstrap robustness: original sparse/dense bins")
lines.append("")
lines.append(
    "This check compares the original graph-degree 0 and 6+ groups. The contrast is the "
    "sparse-group interaction-versus-ego percentage change minus the corresponding dense-group "
    "change, in percentage points. Confidence intervals are track-cluster bootstrap intervals."
)
lines.append("")
lines.append(section_from_candidates(
    "Original-bin paired track bootstrap",
    [
        CONDITION_DIR / "paired_track_bootstrap_200ms.csv",
        CONDITION_DIR / "paired_track_bootstrap_500_1000ms.csv",
        CONDITION_DIR / "paired_track_bootstrap_500ms.csv",
        CONDITION_DIR / "paired_track_bootstrap_1000ms.csv",
    ],
    max_rows=100,
))
lines.append("")
lines.append("## Bootstrap robustness: broader sparse group (0–5) versus dense (6+)")
lines.append("")
lines.append(
    "The broader grouping checks whether the conclusions depend on treating only degree zero as "
    "sparse. It combines degree 0–5 into sparse context and retains degree 6+ as dense context. "
    "The following values are the existing 2,000-replicate track-bootstrap results."
)
lines.append("")
lines.append("| Drone | Horizon (ms) | Target | Sparse change (%) | Dense change (%) | Contrast (pp) | 95% CI (pp) |")
lines.append("|---|---:|---|---:|---:|---:|---|")

broad_path = CONDITION_DIR / "paired_track_bootstrap_sparse0_5_vs_dense6plus.csv"
broad_rows = read_csv(broad_path)
if broad_rows:
    for r in broad_rows:
        def getv(*keys):
            for k in keys:
                if k in r and r[k] not in (None, ""):
                    return r[k]
            return ""
        lines.append(
            "| {drone} | {horizon} | {target} | {sparse} | {dense} | {contrast} | [{lo}, {hi}] |".format(
                drone=getv("drone", "Drone"),
                horizon=getv("horizon_ms", "horizon", "Horizon"),
                target=getv("target", "Target"),
                sparse=getv("sparse_change_pct", "sparse_pct", "sparse_change_percent"),
                dense=getv("dense_change_pct", "dense_pct", "dense_change_percent"),
                contrast=getv("contrast_pp", "contrast_percentage_points", "contrast"),
                lo=getv("ci_low", "ci_lower", "lower_95", "ci95_low"),
                hi=getv("ci_high", "ci_upper", "upper_95", "ci95_high"),
            )
        )
else:
    lines.append(f"| — | — | — | — | — | — | Output not found: `{broad_path.as_posix()}` |")

lines.append("")
lines.append("### Interpretation of the broader-group robustness check")
lines.append("")
lines.append("- Speed: all six sparse-versus-dense contrasts were negative, with 95% confidence intervals below zero.")
lines.append("- Longitudinal acceleration: all six contrasts were negative, with 95% confidence intervals below zero.")
lines.append("- Lateral acceleration: all six contrasts were positive, with 95% confidence intervals above zero.")
lines.append(
    "Thus, relative interaction-history performance is more favorable in the broader sparse group "
    "for speed and longitudinal acceleration, but less favorable for lateral acceleration. This "
    "pattern is consistent across D9 and D10 and all three horizons in the existing bootstrap output."
)
lines.append("")
lines.append("## Limitations")
lines.append("")
lines.append("1. Only two held-out drones are used for the reported test evaluation; transfer to other locations, days, or collection conditions remains untested.")
lines.append("2. Graph degree depends on the chosen 40 m radius and graph construction, and should be described as graph-based local connectivity rather than absolute traffic density.")
lines.append("3. Prediction error differences do not prove that nearby vehicles cause changes in driver behavior.")
lines.append("4. The interaction-history model does not outperform ego-history on every target or context; lateral-acceleration results are a consistent weakness in the reported comparisons.")
lines.append("5. Bootstrap intervals quantify track-resampling uncertainty within the evaluated data; they do not resolve dataset shift or all forms of temporal dependence.")
lines.append("6. The shuffled-history control can create unrealistic input combinations and is best interpreted as a temporal-alignment test.")
lines.append("7. No claims are made that the pNEUMA trajectories provide explicit human preference labels or reveal driver intentions.")
lines.append("")
lines.append("## Conclusions")
lines.append("")
lines.append(
    "The existing experiments support the usefulness of temporally aligned motion history for "
    "trajectory prediction. The shuffled-history control indicates that temporal alignment matters. "
    "The traffic-context analysis suggests that the relative value of neighbor information is "
    "target-dependent: it is more favorable for speed and longitudinal acceleration in sparse "
    "graph contexts than in dense contexts, while lateral acceleration shows the opposite pattern. "
    "These findings should be presented as held-out predictive associations, not causal claims."
)
lines.append("")
lines.append("## Reproducibility and source artifacts")
lines.append("")
for p in [
    "research/human_driving_history_v1.py",
    "research/human_driving_condition_analysis_v1.py",
    "research/human_driving_condition_bootstrap_v1.py",
    "research/human_driving_condition_bootstrap_v2.py",
    "research/human_driving_sparse_dense_contrast_v1.py",
    "research/human_driving_sparse_dense_bootstrap_v2.py",
    "research/human_driving_sparse_dense_bootstrap_v3.py",
]:
    lines.append(f"- `{p}` — {'present' if Path(p).exists() else 'not found in current checkout'}")
lines.append("")
lines.append("Generated from existing artifacts; this report script does not retrain models or create plots.")
lines.append("")

report_path = OUT / "human_driving_history_traffic_context_report.md"
report_path.write_text("\n".join(lines), encoding="utf-8")

manifest = {
    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    "report": report_path.as_posix(),
    "inputs": {
        "history_dir_exists": HISTORY_DIR.exists(),
        "history_ablation_dir_exists": ABLATION_DIR.exists(),
        "condition_dir_exists": CONDITION_DIR.exists(),
        "broad_bootstrap_exists": broad_path.exists(),
    },
    "report_bytes": report_path.stat().st_size,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print(f"Report written: {report_path}")
print(f"Manifest written: {OUT / 'manifest.json'}")
print(f"Report size: {report_path.stat().st_size:,} bytes")
print("\nReview the report for any 'Not found' sections; CSV filenames may differ in your checkout.")
