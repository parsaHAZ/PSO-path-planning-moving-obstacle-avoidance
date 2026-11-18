# pNEUMA Motion-Prediction System Details

> Generated automatically: 2026-10-26T21:03:53+03:30

## 1. Purpose and scope

This document records the computational environment, model implementation, training configuration, graph construction, and evaluation artifacts for the existing pNEUMA motion-prediction experiment. Values are reported from the local environment and repository files where available. Unknown values are marked as unverified rather than inferred.

## 2. Hardware and operating environment

- Operating system/platform: `Windows-10-10.0.19045-SP0`
- Machine architecture: `AMD64`
- Processor reported by Python: `Intel64 Family 6 Model 140 Stepping 1, GenuineIntel`
- Logical CPU count: `8`
- Python executable: `D:\Python310\python.exe`
- Python version: `3.10.0 (tags/v3.10.0:b494f59, Oct  4 2021, 19:00:18) [MSC v.1929 64 bit (AMD64)]`

### System inventory command output

```text
11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
8269307904

Intel(R) Iris(R) Xe Graphics 30.0.101.1404
NVIDIA GeForce MX330         30.0.15.1169
```

### PyTorch and accelerator

- PyTorch: `2.0.1+cpu`
- CUDA available to PyTorch: `False`
- CUDA build reported by PyTorch: `None`
- cuDNN version: `None`
- NumPy: `1.24.3`
- pandas: `1.4.3`
- PyTorch Geometric: `2.4.0`

### GPU devices visible to PyTorch

No CUDA GPU was reported as available to this Python/PyTorch environment. This does not establish whether a GPU is physically installed; see the system inventory output above.

## 3. Dataset split and prediction task

- Dataset: pNEUMA recorded vehicle trajectories.
- Training days: D1–D7.
- Validation day: D8.
- Held-out days: D9–D10, subject to each experiment's eligibility mask.
- Strict benchmark nominal horizon: 80 ms.
- Prediction targets: speed, longitudinal acceleration, and lateral acceleration.
- Reported strict benchmark: D9; D10 is excluded from strict metrics because its target time interval does not meet the configured target-time mask.

These are experiment-level settings; the individual source files below are the authority for exact implementation details.

## 4. Model architecture and training configuration

The model architecture and configuration excerpts below are extracted from the available source files. Tensor shapes are extracted from checkpoint files when their metadata can be loaded safely. A tensor shape alone is not enough to establish every architectural detail.

### `research\validate_and_train_pneuma_80ms.py`

AST parsing unavailable: invalid non-printable character U+FEFF (<unknown>, line 1)

**Relevant source lines**

- L16: `GRAPH_DIR = ROOT / "data" / "pneuma" / "graphs_fixed_80ms"`
- L17: `OUT = ROOT / "artifacts" / "human_driving_validated_80ms"`
- L20: `TRAIN_DAYS = list(range(1, 8))`
- L21: `VAL_DAYS = [8]`
- L22: `TEST_DAYS = [9]`
- L24: `HORIZON_S = 0.080`
- L28: `EPOCHS = 10`
- L29: `PATIENCE = 3`
- L30: `LR = 1e-3`
- L33: `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- L35: `torch.manual_seed(SEED)`
- L48: `data = torch.load(path, map_location="cpu", weights_only=False)`
- L50: `data = torch.load(path, map_location="cpu")`
- L66: `return value.to(dtype=dtype, device="cpu")`
- L70: `required = ("x", "y", "edge_index", "edge_attr", "target_dt")`
- L82: `dt = tensor(graph, "target_dt").reshape(-1)`
- L91: `raise ValueError("x, y and target_dt node counts do not match")`
- L158: `dt = tensor(graph, "target_dt").reshape(-1)`
- L248: `class InteractionGNN(nn.Module):`
- L298: `device=h.device,`
- L307: `device=h.device,`
- L347: `(2, 0), dtype=torch.long, device=DEVICE`
- L350: `(0, 4), dtype=torch.float32, device=DEVICE`
- L446: `optimizer = torch.optim.Adam(model.parameters(), lr=LR)`
- L515: `torch.save(`
- L527: `if stale >= PATIENCE:`
- L554: `train, train_audit = read_days(TRAIN_DAYS, "train")`
- L555: `val, val_audit = read_days(VAL_DAYS, "validation")`
- L556: `test9, test9_audit = read_days(TEST_DAYS, "heldout_D9")`
- L561: `torch.save(`
- L584: `"This run filters saved labels using target_dt and compares "`
- L606: `"interaction_gnn": InteractionGNN(),`

### `research\train_pneuma_human_behavior_v2.py`

AST parsing unavailable: invalid non-printable character U+FEFF (<unknown>, line 1)

**Relevant source lines**

- L15: `GRAPH_DIR = Path("data/pneuma/graphs_fixed_80ms")`
- L16: `OUT_DIR = Path("artifacts/human_driving_fixed80ms_v2")`
- L19: `HORIZON = 0.080`
- L22: `MAX_EPOCHS = 10`
- L23: `PATIENCE = 3`
- L24: `LEARNING_RATE = 1e-3`
- L34: `torch.manual_seed(SEED)`
- L36: `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- L53: `return torch.load(path, map_location="cpu", weights_only=False)`
- L59: `dt = g["target_dt"].float().reshape(-1)`
- L145: `torch.save(stats, stats_path)`
- L174: `class InteractionGNN(nn.Module):`
- L350: `model = InteractionGNN().to(DEVICE)`
- L352: `model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4`
- L382: `torch.save({`
- L393: `if stale >= PATIENCE:`
- L404: `ckpt = torch.load(`

## 5. Preprocessing and graph construction

### `research\build_pneuma_all_vehicle_graphs_80ms.py`

AST parsing unavailable: invalid non-printable character U+FEFF (<unknown>, line 1)

**Relevant source lines**

- L13: `OUTPUT_DIR = ROOT / "data" / "pneuma" / "graphs_all_vehicles_80ms"`
- L16: `HORIZON_S = 0.080`
- L18: `RADIUS_M = 40.0`
- L52: `NaN targets and NaN target_dt.`
- L133: `Build directed edges for all pairs within the 40 m radius.`
- L146: `pairs = tree.query_pairs(RADIUS_M, output_type="ndarray")`
- L159: `keep = (distance > 0.0) & (distance <= RADIUS_M)`
- L200: `"builder": "build_pneuma_all_vehicle_graphs_80ms.py",`
- L205: `"radius_m": RADIUS_M,`
- L238: `df["_target_dt"] = all_dt`
- L274: `target_dt = frame["_target_dt"].to_numpy(dtype=np.float32)`
- L286: `"target_dt": torch.from_numpy(target_dt.copy()),`
- L295: `matched_targets += int(np.isfinite(target_dt).sum())`
- L297: `# Do not overwrite anything in graphs_fixed_80ms.`
- L298: `torch.save(graphs, output_path)`

### `research\pneuma_preprocess.py`

**Classes**
- No top-level classes found.

**Functions**
- Function `read_pneuma_wide` (line 14)
- Function `add_local_coordinates` (line 112)
- Function `add_derived_features` (line 132)
- Function `preprocess_file` (line 158)
- Function `main` (line 175)

**Relevant configuration assignments**
- No relevant top-level assignments found.

**Relevant source lines**

- L118: `earth_radius = 6_371_000.0`
- L126: `out["x_m"] = earth_radius * (lon - lon0) * np.cos(lat0)`
- L127: `out["y_m"] = earth_radius * (lat - np.deg2rad(out["lat"].mean()))`

## 6. Existing checkpoint inspection

### `artifacts\human_driving_validated_80ms\best_interaction_gnn.pt`

- File size: 226,759 bytes
- Safe metadata load failed: `UnpicklingError: Weights only load failed. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution.Do it only if you get the file from a trusted source. WeightsUnpickler error: Unsupported class numpy.core.multiarray.scalar`
- No unrestricted pickle load was attempted.

### `artifacts\gnn\20181101_multidrone.pt`

- File size: 228,254 bytes
- Loaded with `weights_only=True`: yes
- Top-level keys: model_state_dict, node_dim, edge_dim, hidden_dim, out_dim, target_mean, target_std, train_drones, validation_drones, test_drones, test_mae, per_drone_test_mae, split_type, date, time_window, seed
- State dictionary located under `model_state_dict`
- Tensor shapes:
  - `node_encoder.0.weight`: (64, 7)
  - `node_encoder.0.bias`: (64,)
  - `node_encoder.2.weight`: (64, 64)
  - `node_encoder.2.bias`: (64,)
  - `edge_encoder.0.weight`: (64, 4)
  - `edge_encoder.0.bias`: (64,)
  - `edge_encoder.2.weight`: (64, 64)
  - `edge_encoder.2.bias`: (64,)
  - `message_mlp.0.weight`: (64, 192)
  - `message_mlp.0.bias`: (64,)
  - `message_mlp.2.weight`: (64, 64)
  - `message_mlp.2.bias`: (64,)
  - `update.weight_ih`: (192, 64)
  - `update.weight_hh`: (192, 64)
  - `update.bias_ih`: (192,)
  - `update.bias_hh`: (192,)
  - `prediction_head.0.weight`: (64, 64)
  - `prediction_head.0.bias`: (64,)
  - `prediction_head.2.weight`: (3, 64)
  - `prediction_head.2.bias`: (3,)
- `seed`: `42`

## 7. Evaluation artifacts and statistical procedure

- Row-level metrics: `artifacts/human_driving_validated_80ms/metrics.csv`
- Equal-track metrics: `artifacts/human_driving_validated_80ms/paired_track_metrics.csv`
- Paired bootstrap: `artifacts/human_driving_validated_80ms/paired_track_bootstrap.csv`
- Data audit: `artifacts/human_driving_validated_80ms/data_audit.json`
- Paired predictions: `artifacts/human_driving_validated_80ms/heldout_paired_predictions.csv`

The existing track-level bootstrap uses 2,000 resamples and seed 20261023, as recorded for the current evaluation. The saved evaluation artifacts were previously checked for duplicate keys, non-finite predictions, track-metric reconciliation, and bootstrap reproducibility. These checks concern the existing artifacts and do not independently prove that all artifacts were generated from the corrected preprocessing source.

## 8. Provenance and known limitations

1. The heading preprocessing source was corrected to avoid using future observations to fill a missing first-observation heading. The existing processed CSVs, graph files, checkpoints, and predictions were not regenerated by that source edit. The existing metrics must therefore be identified as results from the prior artifacts, provisional with respect to this correction.
2. The strict graph builder and training script have been observed to reference different graph directories (`graphs_all_vehicles_80ms` versus `graphs_fixed_80ms`). The exact provenance of the checkpoint's input graphs must not be asserted solely from the current script names.
3. Hardware details in this report describe the environment at the time this file was generated. They do not prove that the recorded training run used a particular GPU unless the training logs or run metadata establish that fact.
4. Any unavailable parameter counts, package versions, or configuration values should remain unreported until verified from the relevant source or logs.

## 9. Recommended citation in the paper

Use this document as the source for the system-configuration subsection. Cite the exact model settings from the source code and checkpoint, and clearly distinguish the current inspection environment from the environment used for the original training run if they differ.
