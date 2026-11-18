
from pathlib import Path
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(".")
GRAPH_DIR = ROOT / "data" / "pneuma" / "graphs_fixed_80ms"
OUT = ROOT / "artifacts" / "human_driving_validated_80ms"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_DAYS = list(range(1, 8))
VAL_DAYS = [8]
TEST_DAYS = [9]

HORIZON_S = 0.080
TOLERANCE_S = 0.001

BATCH_GRAPHS = 32
EPOCHS = 10
PATIENCE = 3
LR = 1e-3
SEED = 2026

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_num_threads(1)


def load_day(day):
    path = GRAPH_DIR / f"20181101_d{day}_0800_0830.pt"

    if not path.exists():
        raise FileNotFoundError(f"Missing graph file: {path}")

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")

    if not isinstance(data, (list, tuple)):
        raise TypeError(
            f"{path}: expected a list/tuple of graphs, got {type(data)}"
        )

    return list(data)


def tensor(graph, key, dtype=torch.float32):
    value = graph[key]

    if not torch.is_tensor(value):
        value = torch.as_tensor(value)

    return value.to(dtype=dtype, device="cpu")


def unpack(graph):
    required = ("x", "y", "edge_index", "edge_attr", "target_dt")
    missing = [key for key in required if key not in graph]

    if missing:
        raise KeyError(
            f"Graph missing fields {missing}; available={list(graph.keys())}"
        )

    x = tensor(graph, "x")
    y = tensor(graph, "y")
    edge_index = tensor(graph, "edge_index", torch.long)
    edge_attr = tensor(graph, "edge_attr")
    dt = tensor(graph, "target_dt").reshape(-1)

    if x.ndim != 2 or x.shape[1] != 7:
        raise ValueError(f"Expected x shape (N,7), got {tuple(x.shape)}")

    if y.ndim != 2 or y.shape[1] != 3:
        raise ValueError(f"Expected y shape (N,3), got {tuple(y.shape)}")

    if dt.numel() != x.shape[0] or y.shape[0] != x.shape[0]:
        raise ValueError("x, y and target_dt node counts do not match")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index shape (2,E), got {tuple(edge_index.shape)}"
        )

    if edge_attr.ndim != 2 or edge_attr.shape != (
        edge_index.shape[1], 4
    ):
        raise ValueError(
            "edge_attr must have shape (E,4) matching edge_index"
        )

    # Only finite targets whose recorded target interval is 80 ms +/- 1 ms.
    valid = (
        torch.isfinite(x).all(dim=1)
        & torch.isfinite(y).all(dim=1)
        & torch.isfinite(dt)
        & ((dt - HORIZON_S).abs() <= TOLERANCE_S)
    )

    if edge_index.numel():
        edge_ok = (
            (edge_index >= 0).all()
            and (edge_index < x.shape[0]).all()
            and torch.isfinite(edge_attr).all()
        )

        if not bool(edge_ok):
            raise ValueError(
                "Invalid edge indices or non-finite edge features"
            )

    return x, y, edge_index, edge_attr, valid


def read_days(days, name):
    result = []

    audit = {
        "split": name,
        "days": list(days),
        "graphs": 0,
        "nodes": 0,
        "valid_nodes": 0,
        "valid_dt_min": None,
        "valid_dt_median": None,
        "valid_dt_max": None,
    }

    dts = []

    for day in days:
        graphs = load_day(day)
        day_valid = 0
        day_nodes = 0

        for graph in graphs:
            x, y, edge_index, edge_attr, mask = unpack(graph)

            result.append((x, y, edge_index, edge_attr, mask))

            day_nodes += x.shape[0]
            day_valid += int(mask.sum())

            if bool(mask.any()):
                dt = tensor(graph, "target_dt").reshape(-1)
                dts.append(dt[mask].numpy())

        audit["graphs"] += len(graphs)
        audit["nodes"] += day_nodes
        audit["valid_nodes"] += day_valid

        print(
            f"{name}: D{day}: graphs={len(graphs):,}, "
            f"nodes={day_nodes:,}, valid target nodes={day_valid:,}",
            flush=True,
        )

    if not result:
        raise RuntimeError(f"No graphs loaded for {name}")

    if audit["valid_nodes"] == 0:
        raise RuntimeError(f"No valid target nodes in {name}")

    if dts:
        values = np.concatenate(dts)
        audit["valid_dt_min"] = float(values.min())
        audit["valid_dt_median"] = float(np.median(values))
        audit["valid_dt_max"] = float(values.max())

    print(f"{name} audit: {json.dumps(audit)}", flush=True)

    return result, audit


def feature_stats(graphs):
    # Normalization statistics are fitted on training data only.
    xs, ys, es = [], [], []

    for x, y, edge_index, edge_attr, mask in graphs:
        if bool(mask.any()):
            xs.append(x[mask])
            ys.append(y[mask])

        # Edge features are observed inputs; finite training edges are used.
        if edge_attr.numel():
            finite_edges = torch.isfinite(edge_attr).all(dim=1)
            if bool(finite_edges.any()):
                es.append(edge_attr[finite_edges])

    if not xs or not ys:
        raise RuntimeError(
            "Could not calculate training normalization statistics"
        )

    X = torch.cat(xs)
    Y = torch.cat(ys)

    if es:
        E = torch.cat(es)
    else:
        E = torch.zeros((0, 4))

    def stats(values, fallback_dim):
        if values.numel() == 0:
            return torch.zeros(fallback_dim), torch.ones(fallback_dim)

        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp_min(1e-6)

        return mean, std

    xm, xsig = stats(X, 7)
    ym, ysig = stats(Y, 3)
    em, esig = stats(E, 4)

    return xm, xsig, ym, ysig, em, esig


class EgoMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.net(x)


class InteractionGNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.edge_encoder = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.message = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.update = nn.GRUCell(hidden, hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.node_encoder(x)

        if edge_index.shape[1] == 0:
            return self.head(h)

        src, dst = edge_index
        e = self.edge_encoder(edge_attr)

        msg = self.message(
            torch.cat([h[dst], h[src], e], dim=1)
        )

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)

        degree = torch.zeros(
            (h.shape[0], 1),
            dtype=h.dtype,
            device=h.device,
        )

        degree.index_add_(
            0,
            dst,
            torch.ones(
                (dst.numel(), 1),
                dtype=h.dtype,
                device=h.device,
            ),
        )

        agg = agg / degree.clamp_min(1.0)
        h = self.update(agg, h)

        return self.head(h)


def normalized_graph(batch_graphs, stats):
    xm, xsig, ym, ysig, em, esig = stats

    X, Y, masks = [], [], []
    edge_indices, edge_attrs = [], []

    offset = 0

    for x, y, edge_index, edge_attr, mask in batch_graphs:
        n = x.shape[0]

        X.append((x - xm) / xsig)
        Y.append((y - ym) / ysig)
        masks.append(mask)

        if edge_index.numel():
            edge_indices.append(edge_index + offset)
            edge_attrs.append((edge_attr - em) / esig)

        offset += n

    X = torch.cat(X).to(DEVICE)
    Y = torch.cat(Y).to(DEVICE)
    mask = torch.cat(masks).to(DEVICE)

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1).to(DEVICE)
        edge_attr = torch.cat(edge_attrs, dim=0).to(DEVICE)
    else:
        edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=DEVICE
        )
        edge_attr = torch.empty(
            (0, 4), dtype=torch.float32, device=DEVICE
        )

    return X, Y, mask, edge_index, edge_attr


def batches(graphs, shuffle=False):
    order = list(range(len(graphs)))

    if shuffle:
        random.shuffle(order)

    for start in range(0, len(order), BATCH_GRAPHS):
        yield [graphs[i] for i in order[start:start + BATCH_GRAPHS]]


@torch.no_grad()
def evaluate(model, graphs, stats, name):
    model.eval()

    _, _, ym, ysig, _, _ = stats

    sums_abs = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    count = 0

    for batch_graphs in batches(graphs):
        X, Y, mask, edge_index, edge_attr = normalized_graph(
            batch_graphs, stats
        )

        pred_norm = model(X, edge_index, edge_attr)

        pred = pred_norm * ysig.to(DEVICE) + ym.to(DEVICE)
        target = Y * ysig.to(DEVICE) + ym.to(DEVICE)

        if not bool(mask.any()):
            continue

        err = (pred[mask] - target[mask]).double()

        sums_abs += err.abs().sum(dim=0).cpu().numpy()
        sums_sq += err.square().sum(dim=0).cpu().numpy()
        count += int(mask.sum())

    if count == 0:
        raise RuntimeError(f"No evaluation targets for {name}")

    return {
        "split": name,
        "n": count,
        "speed_mae": sums_abs[0] / count,
        "lon_acc_mae": sums_abs[1] / count,
        "lat_acc_mae": sums_abs[2] / count,
        "speed_rmse": np.sqrt(sums_sq[0] / count),
        "lon_acc_rmse": np.sqrt(sums_sq[1] / count),
        "lat_acc_rmse": np.sqrt(sums_sq[2] / count),
    }


@torch.no_grad()
def evaluate_persistence(graphs, name):
    sums_abs = np.zeros(3, dtype=np.float64)
    sums_sq = np.zeros(3, dtype=np.float64)
    count = 0

    for x, y, edge_index, edge_attr, mask in graphs:
        if not bool(mask.any()):
            continue

        # Predict next speed and accelerations using current values.
        pred = x[:, [2, 3, 4]]
        err = (pred[mask] - y[mask]).double()

        sums_abs += err.abs().sum(dim=0).numpy()
        sums_sq += err.square().sum(dim=0).numpy()
        count += int(mask.sum())

    if count == 0:
        raise RuntimeError(f"No persistence targets for {name}")

    return {
        "model": "persistence",
        "split": name,
        "n": count,
        "speed_mae": sums_abs[0] / count,
        "lon_acc_mae": sums_abs[1] / count,
        "lat_acc_mae": sums_abs[2] / count,
        "speed_rmse": np.sqrt(sums_sq[0] / count),
        "lon_acc_rmse": np.sqrt(sums_sq[1] / count),
        "lat_acc_rmse": np.sqrt(sums_sq[2] / count),
    }


def train_model(model_name, model, train, val, stats):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        model.train()

        loss_sum = 0.0
        node_count = 0

        for batch_graphs in batches(train, shuffle=True):
            X, Y, mask, edge_index, edge_attr = normalized_graph(
                batch_graphs, stats
            )

            if not bool(mask.any()):
                continue

            pred = model(X, edge_index, edge_attr)
            loss = F.smooth_l1_loss(pred[mask], Y[mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            n = int(mask.sum())
            loss_sum += float(loss.detach()) * n
            node_count += n

        val_metrics = evaluate(model, val, stats, "D8")

        # Retain the existing validation scoring rule for comparability.
        val_score = (
            val_metrics["speed_mae"] / 4.28055429
            + val_metrics["lon_acc_mae"] / 0.77070218
            + val_metrics["lat_acc_mae"] / 0.36888441
        ) / 3.0

        row = {
            "model": model_name,
            "epoch": epoch,
            "train_smooth_l1": loss_sum / max(node_count, 1),
            "val_normalized_mae_score": val_score,
            **val_metrics,
        }

        history.append(row)

        print(
            f"{model_name} epoch {epoch}/{EPOCHS}: "
            f"train_loss={row['train_smooth_l1']:.6f}, "
            f"D8 speed/lon/lat MAE="
            f"{val_metrics['speed_mae']:.6f}/"
            f"{val_metrics['lon_acc_mae']:.6f}/"
            f"{val_metrics['lat_acc_mae']:.6f}, "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

        if val_score < best_val:
            best_val = val_score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0

            torch.save(
                {
                    "model_name": model_name,
                    "state_dict": best_state,
                    "epoch": epoch,
                    "val_score": best_val,
                },
                OUT / f"best_{model_name}.pt",
            )
        else:
            stale += 1

            if stale >= PATIENCE:
                print(f"{model_name}: early stopping", flush=True)
                break

    if best_state is None:
        raise RuntimeError(
            f"No best checkpoint produced for {model_name}"
        )

    model.load_state_dict(best_state)

    pd.DataFrame(history).to_csv(
        OUT / f"history_{model_name}.csv",
        index=False,
    )

    return model


def main():
    print(f"Device: {DEVICE}")
    print(f"Graph directory: {GRAPH_DIR.resolve()}")
    print(f"Output directory: {OUT.resolve()}")
    print(f"Required target interval: {HORIZON_S:.3f} "
          f"+/- {TOLERANCE_S:.3f} seconds")
    print("Loading splits...", flush=True)

    train, train_audit = read_days(TRAIN_DAYS, "train")
    val, val_audit = read_days(VAL_DAYS, "validation")
    test9, test9_audit = read_days(TEST_DAYS, "heldout_D9")

    stats = feature_stats(train)

    # Save training-only normalization statistics.
    torch.save(
        {
            "x_mean": stats[0],
            "x_std": stats[1],
            "y_mean": stats[2],
            "y_std": stats[3],
            "edge_mean": stats[4],
            "edge_std": stats[5],
        },
        OUT / "normalization.pt",
    )

    audit = {
        "horizon_s": HORIZON_S,
        "tolerance_s": TOLERANCE_S,
        "split_policy": (
            "D1-D7 train, D8 validation, D9 heldout test; "
            "D10 excluded because its recorded target interval "
            "is approximately 83.417 ms."
        ),
        "training_only_normalization": True,
        "audits": [train_audit, val_audit, test9_audit],
        "important_limit": (
            "This run filters saved labels using target_dt and compares "
            "predictors at the selected horizon. It does not independently "
            "prove how y was constructed from raw trajectories or establish "
            "that every input feature is free from future information."
        ),
    }

    (OUT / "data_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    rows = []

    # Baseline evaluation: validation and strict-horizon heldout D9 only.
    for name, split in [("D8", val), ("D9", test9)]:
        result = evaluate_persistence(split, name)
        rows.append(result)
        print(f"Persistence {name}: {result}", flush=True)

    models = {
        "ego_mlp": EgoMLP(),
        "interaction_gnn": InteractionGNN(),
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...", flush=True)

        fitted = train_model(
            model_name,
            model,
            train,
            val,
            stats,
        )

        for name, split in [("D8", val), ("D9", test9)]:
            metrics = evaluate(fitted, split, stats, name)
            metrics["model"] = model_name
            rows.append(metrics)

            print(
                f"{model_name} {name}: {metrics}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "metrics.csv", index=False)

    print("\nFINAL METRICS")
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT / 'metrics.csv'}")
    print(f"Saved: {OUT / 'data_audit.json'}")
    print("\nD10 was not loaded or evaluated in this run.")


if __name__ == "__main__":
    main()
