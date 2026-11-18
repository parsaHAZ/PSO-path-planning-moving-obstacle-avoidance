from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gnn_behavior import InteractionGNN


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_drone_graphs(graph_dir: str, drone_ids):
    graph_dir = Path(graph_dir)
    all_graphs = []

    for drone_id in drone_ids:
        filename = f"20181101_d{drone_id}_0800_0830.pt"
        path = graph_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing graph file:\n{path}")

        print(f"Loading Drone {drone_id}: {path}")
        graphs = torch.load(path, map_location="cpu")
        print(f"  Graphs: {len(graphs):,}")

        for graph in graphs:
            graph = dict(graph)
            graph["drone_id"] = drone_id
            all_graphs.append(graph)

    return all_graphs


def calculate_target_statistics(graphs):
    targets = []
    for graph in graphs:
        y = graph["y"]
        valid = torch.isfinite(y).all(dim=1)
        if valid.any():
            targets.append(y[valid])

    if not targets:
        raise RuntimeError("No valid targets found.")

    targets = torch.cat(targets, dim=0)
    mean = targets.mean(dim=0)
    std = targets.std(dim=0)
    std = torch.clamp(std, min=1e-8)

    return mean, std


def normalize_targets(y, mean, std):
    return (y - mean) / std


def get_model_prediction(model, x, edge_index, edge_attr):
    output = model(x, edge_index, edge_attr)

    # Supports either: prediction or (prediction, embedding)
    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError(f"InteractionGNN returned unsupported type: {type(output)}")

    return output


@torch.no_grad()
def evaluate(model, graphs, target_mean, target_std, device):
    model.eval()

    total_absolute_error = torch.zeros(3, device=device)
    total_count = 0

    for graph in graphs:
        x = graph["x"].float().to(device)
        edge_index = graph["edge_index"].long().to(device)
        edge_attr = graph["edge_attr"].float().to(device)
        y = graph["y"].float().to(device)

        valid = torch.isfinite(y).all(dim=1)
        if valid.sum() == 0:
            continue

        prediction = get_model_prediction(model, x, edge_index, edge_attr)
        prediction = prediction[valid]
        y = y[valid]

        prediction_physical = prediction * target_std.to(device) + target_mean.to(device)
        error = torch.abs(prediction_physical - y)

        total_absolute_error += error.sum(dim=0)
        total_count += y.shape[0]

    if total_count == 0:
        return None

    return (total_absolute_error / total_count).cpu()


@torch.no_grad()
def evaluate_per_drone(model, graphs, target_mean, target_std, device):
    drone_results = {}
    drone_ids = sorted(set(graph["drone_id"] for graph in graphs))

    for drone_id in drone_ids:
        drone_graphs = [graph for graph in graphs if graph["drone_id"] == drone_id]
        mae = evaluate(model, drone_graphs, target_mean, target_std, device)
        if mae is not None:
            drone_results[drone_id] = mae

    return drone_results


def train(model, train_graphs, val_graphs, target_mean, target_std, device, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val = float("inf")
    best_state = None

    target_mean_device = target_mean.to(device)
    target_std_device = target_std.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_graphs)

        total_loss = 0.0
        graph_count = 0

        for graph in train_graphs:
            x = graph["x"].float().to(device)
            edge_index = graph["edge_index"].long().to(device)
            edge_attr = graph["edge_attr"].float().to(device)
            y = graph["y"].float().to(device)

            valid = torch.isfinite(y).all(dim=1)
            if valid.sum() == 0:
                continue

            optimizer.zero_grad()

            prediction = get_model_prediction(model, x, edge_index, edge_attr)
            prediction = prediction[valid]
            y = y[valid]

            target = normalize_targets(y, target_mean_device, target_std_device)
            loss = F.smooth_l1_loss(prediction, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            graph_count += 1

        val_mae = evaluate(model, val_graphs, target_mean, target_std, device)
        if val_mae is None:
            continue

        mean_val_mae = val_mae.mean().item()

        if mean_val_mae < best_val:
            best_val = mean_val_mae
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0:
            train_loss = total_loss / max(graph_count, 1)
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Validation MAE: {mean_val_mae:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", default="data/pneuma/graphs")
    parser.add_argument("--output", default="artifacts/gnn/20181101_multidrone.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Drone split: D1-D7 train, D8 validation, D9-D10 test
    train_drones = list(range(1, 8))
    validation_drones = [8]
    test_drones = [9, 10]

    print("\nDrone split:")
    print(f"Train: D{train_drones[0]}-D{train_drones[-1]}")
    print(f"Validation : D8")
    print(f"Test: D9-D10")

    train_graphs = load_drone_graphs(args.graph_dir, train_drones)
    val_graphs = load_drone_graphs(args.graph_dir, validation_drones)
    test_graphs = load_drone_graphs(args.graph_dir, test_drones)

    print("\nDataset sizes:")
    print(f"Train graphs: {len(train_graphs):,}")
    print(f"Validation graphs: {len(val_graphs):,}")
    print(f"Test graphs: {len(test_graphs):,}")

    # Statistics from training drones only
    target_mean, target_std = calculate_target_statistics(train_graphs)
    print("\nTraining target statistics:")
    print(f"Mean: {target_mean.numpy()}")
    print(f"Std : {target_std.numpy()}")

    model = InteractionGNN(
        node_dim=7,
        edge_dim=4,
        hidden_dim=64,
        out_dim=3,
    ).to(device)

    print("\nTraining GNN...")

    train(
        model=model,
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
        epochs=args.epochs,
    )

    test_mae = evaluate(
        model=model,
        graphs=test_graphs,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
    )
    if test_mae is None:
        raise RuntimeError("No valid test predictions.")

    per_drone = evaluate_per_drone(
        model=model,
        graphs=test_graphs,
        target_mean=target_mean,
        target_std=target_std,
        device=device,
    )

    print("\n" + "=" * 70)
    print("MULTI-DRONE TEST RESULTS")
    print("=" * 70)
    print("Test drones: D9, D10")
    print("\nOverall:")
    print(f"Next speed MAE: {test_mae[0].item():.6f}")
    print(f"Longitudinal acc MAE: {test_mae[1].item():.6f}")
    print(f"Lateral acc MAE: {test_mae[2].item():.6f}")
    print(f"Overall MAE: {test_mae.mean().item():.6f}")

    print("\n" + "-" * 70)
    print("PER-DRONE TEST RESULTS")
    print("-" * 70)

    for drone_id in sorted(per_drone.keys()):
        mae = per_drone[drone_id]
        print(f"Drone {drone_id:02d} | Speed: {mae[0].item():.6f} | Lon Acc: {mae[1].item():.6f} | Lat Acc: {mae[2].item():.6f} | Overall: {mae.mean().item():.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_dim": 7,
            "edge_dim": 4,
            "hidden_dim": 64,
            "out_dim": 3,
            "target_mean": target_mean,
            "target_std": target_std,
            "train_drones": train_drones,
            "validation_drones": validation_drones,
            "test_drones": test_drones,
            "test_mae": test_mae,
            "per_drone_test_mae": per_drone,
            "split_type": "drone_disjoint",
            "date": "2018-11-01",
            "time_window": "08:00-08:30",
            "seed": args.seed,
        },
        output_path,
    )

    print(f"\nCheckpoint saved to: {output_path}")


if __name__ == "__main__":
    main()