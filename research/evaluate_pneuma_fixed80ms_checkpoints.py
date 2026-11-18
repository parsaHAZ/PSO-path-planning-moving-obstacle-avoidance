from pathlib import Path
import csv
import gc
import math
import time

import torch
from torch import nn

ROOT = Path("artifacts/human_driving_fixed80ms_v1")
GRAPH_DIR = Path("data/pneuma/graphs_fixed_80ms")
OUT = ROOT / "heldout_metrics.csv"

HORIZON = 0.080
TOLERANCE = 0.0041
FEATURES = ("speed_mps", "longitudinal_acceleration", "lateral_acceleration")

torch.set_num_threads(1)


class EgoMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.net(x)


class InteractionGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(7, 64), nn.ReLU())
        self.message1 = nn.Sequential(
            nn.Linear(132, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.update1 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.message2 = nn.Sequential(
            nn.Linear(132, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.update2 = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 3),
        )

    @staticmethod
    def aggregate(messages, dst, n_nodes):
        result = messages.new_zeros((n_nodes, messages.shape[-1]))
        degree = messages.new_zeros((n_nodes, 1))
        if dst.numel():
            result.index_add_(0, dst, messages)
            degree.index_add_(
                0, dst, messages.new_ones((dst.numel(), 1))
            )
        return result / degree.clamp_min(1.0)

    def forward(self, x, edge_index, edge_attr):
        h = self.encoder(x)

        src, dst = edge_index[0], edge_index[1]
        if src.numel():
            msg = self.message1(
                torch.cat((h[src], h[dst], edge_attr), dim=1)
            )
            agg = self.aggregate(msg, dst, h.shape[0])
        else:
            agg = torch.zeros_like(h)
        h = h + self.update1(torch.cat((h, agg), dim=1))

        if src.numel():
            msg = self.message2(
                torch.cat((h[src], h[dst], edge_attr), dim=1)
            )
            agg = self.aggregate(msg, dst, h.shape[0])
        else:
            agg = torch.zeros_like(h)
        h = h + self.update2(torch.cat((h, agg), dim=1))

        return self.head(h)


def load_model(filename, model_class):
    ckpt = torch.load(ROOT / filename, map_location="cpu", weights_only=False)
    model = model_class()
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    stats = {}
    for key in ("x_mean", "x_std", "y_mean", "y_std", "e_mean", "e_std"):
        stats[key] = ckpt[key].float().reshape(1, -1)

    print(
        f"Loaded {filename}: epoch={ckpt['best_epoch']}, "
        f"validation_normalized_mse={ckpt['validation_normalized_mse']:.8f}"
    )
    return model, stats


def make_safe_normalized(raw, mean, std):
    raw = raw.float()
    safe = torch.where(torch.isfinite(raw), raw, mean.expand_as(raw))
    return (safe - mean) / std.clamp_min(1e-6)


def new_accumulator():
    return {
        "count": [0, 0, 0],
        "abs_sum": [0.0, 0.0, 0.0],
        "sq_sum": [0.0, 0.0, 0.0],
    }


def add_errors(acc, pred, target, mask):
    for j in range(3):
        valid = mask & torch.isfinite(pred[:, j]) & torch.isfinite(target[:, j])
        if not valid.any():
            continue
        err = (pred[valid, j] - target[valid, j]).double()
        acc["count"][j] += int(err.numel())
        acc["abs_sum"][j] += err.abs().sum().item()
        acc["sq_sum"][j] += err.square().sum().item()


def metric_rows(acc, day):
    rows = []
    for model_name, values in acc.items():
        for j, feature in enumerate(FEATURES):
            n = values["count"][j]
            rows.append({
                "day": day,
                "model": model_name,
                "target": feature,
                "count": n,
                "MAE": values["abs_sum"][j] / n if n else float("nan"),
                "RMSE": math.sqrt(values["sq_sum"][j] / n) if n else float("nan"),
            })
    return rows


def evaluate_day(day, models):
    path = GRAPH_DIR / f"20181101_d{day}_0800_0830.pt"
    if not path.exists():
        raise FileNotFoundError(path)

    print(f"\nLoading D{day}: {path} ({path.stat().st_size / 1e9:.2f} GB)")
    graphs = torch.load(path, map_location="cpu", weights_only=False)
    print(f"D{day}: {len(graphs):,} graph snapshots")

    acc = {
        "persistence": new_accumulator(),
        "ego_mlp": new_accumulator(),
        "interaction_gnn": new_accumulator(),
    }

    start = time.time()
    eligible_total = 0

    with torch.inference_mode():
        for i, graph in enumerate(graphs):
            x_raw = graph["x"].float()
            y_raw = graph["y"].float()
            dt = graph["target_dt"].float().reshape(-1)

            mask = (
                torch.isfinite(y_raw).all(dim=1)
                & torch.isfinite(x_raw).all(dim=1)
                & torch.isfinite(dt)
                & ((dt - HORIZON).abs() <= TOLERANCE)
            )

            if not mask.any():
                continue

            eligible_total += int(mask.sum())

            # Persistence: current speed and accelerations predict their
            # respective values at the fixed future horizon.
            add_errors(acc["persistence"], x_raw[:, 2:5], y_raw, mask)

            edge_index = graph["edge_index"].long()
            edge_raw = graph["edge_attr"].float()

            for model_name in ("ego_mlp", "interaction_gnn"):
                model, stats = models[model_name]
                x_norm = make_safe_normalized(
                    x_raw, stats["x_mean"], stats["x_std"]
                )

                if model_name == "interaction_gnn":
                    edge_norm = make_safe_normalized(
                        edge_raw, stats["e_mean"], stats["e_std"]
                    )
                else:
                    edge_norm = edge_raw

                pred_norm = model(x_norm, edge_index, edge_norm)
                pred_raw = pred_norm * stats["y_std"] + stats["y_mean"]
                add_errors(acc[model_name], pred_raw, y_raw, mask)

            if (i + 1) % 2000 == 0:
                print(
                    f"  D{day}: {i+1:,}/{len(graphs):,} snapshots; "
                    f"{eligible_total:,} eligible targets; "
                    f"{time.time()-start:.0f}s elapsed",
                    flush=True,
                )

    print(f"D{day}: evaluation finished in {time.time()-start:.1f}s")
    print(f"D{day}: {eligible_total:,} eligible node targets")
    rows = metric_rows(acc, f"D{day}")

    for row in rows:
        print(
            f"  {row['model']:16s} {row['target']:27s} "
            f"n={row['count']:>9,} "
            f"MAE={row['MAE']:.6f} RMSE={row['RMSE']:.6f}"
        )

    del graphs
    gc.collect()
    return rows, acc


def main():
    models = {
        "ego_mlp": load_model("best_ego_mlp.pt", EgoMLP),
        "interaction_gnn": load_model(
            "best_interaction_gnn.pt", InteractionGNN
        ),
    }

    all_rows = []
    pooled = {
        name: new_accumulator()
        for name in ("persistence", "ego_mlp", "interaction_gnn")
    }

    for day in (9, 10):
        rows, day_acc = evaluate_day(day, models)
        all_rows.extend(rows)

        for model_name, values in day_acc.items():
            for j in range(3):
                pooled[model_name]["count"][j] += values["count"][j]
                pooled[model_name]["abs_sum"][j] += values["abs_sum"][j]
                pooled[model_name]["sq_sum"][j] += values["sq_sum"][j]

    all_rows.extend(metric_rows(pooled, "D9+D10"))

    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["day", "model", "target", "count", "MAE", "RMSE"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n=== POOLED HELD-OUT RESULTS: D9 + D10 ===")
    for row in all_rows:
        if row["day"] == "D9+D10":
            print(
                f"{row['model']:16s} {row['target']:27s} "
                f"n={row['count']:>9,} "
                f"MAE={row['MAE']:.6f} RMSE={row['RMSE']:.6f}"
            )

    print(f"\nSaved metrics: {OUT}")


if __name__ == "__main__":
    main()
