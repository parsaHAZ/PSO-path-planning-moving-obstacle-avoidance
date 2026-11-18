from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import nn


class InteractionGNN(nn.Module):
    def __init__(self, node_dim=7, edge_dim=4, hidden_dim=64, out_dim=3):
        super().__init__()

        # Encode each vehicle's own state
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Encode interaction information
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message construction: source + destination + interaction
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update each node after receiving messages
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

        # Predict next vehicle behavior
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        # Initial vehicle representations
        h = self.node_encoder(x)

        # Edge representations
        e = self.edge_encoder(edge_attr)

        src, dst = edge_index

        # Construct messages
        messages = self.message_mlp(torch.cat([h[src], h[dst], e], dim=-1))

        # Aggregate messages at destination nodes
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, messages)

        # Update node representations
        h = self.update(aggregated, h)

        # Predict future behavior
        prediction = self.prediction_head(h)

        return prediction, h


def split_graphs(graphs, train_ratio=0.70, validation_ratio=0.15):
    n = len(graphs)

    n_train = max(1, int(n * train_ratio))
    n_validation = max(1, int(n * validation_ratio))

    # Keep at least one test graph
    n_train = min(n_train, n - 2)
    n_validation = min(n_validation, n - n_train - 1)

    train = graphs[:n_train]
    validation = graphs[n_train:n_train + n_validation]
    test = graphs[n_train + n_validation:]

    return train, validation, test


def calculate_target_statistics(graphs):
    y = torch.cat([graph["y"] for graph in graphs], dim=0)
    mean = y.mean(dim=0)
    std = y.std(dim=0).clamp_min(1e-6)
    return mean, std


def evaluate(model, graphs, target_mean, target_std, device):
    model.eval()

    absolute_error = 0.0
    number_of_nodes = 0

    with torch.no_grad():
        for graph in graphs:
            x = graph["x"].to(device)
            edge_index = graph["edge_index"].to(device)
            edge_attr = graph["edge_attr"].to(device)
            y = graph["y"].to(device)

            prediction, _ = model(x, edge_index, edge_attr)

            # Convert normalized prediction back to physical units
            prediction = prediction * target_std + target_mean
            error = torch.abs(prediction - y)

            absolute_error += error.sum().item()
            number_of_nodes += error.numel()

    return absolute_error / max(number_of_nodes, 1)


def train(graphs, epochs=30, learning_rate=1e-3, hidden_dim=64, seed=42):
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_graphs, validation_graphs, test_graphs = split_graphs(graphs)

    target_mean, target_std = calculate_target_statistics(train_graphs)
    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    model = InteractionGNN(
        node_dim=7,
        edge_dim=4,
        hidden_dim=hidden_dim,
        out_dim=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.SmoothL1Loss()

    best_validation_mae = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_nodes = 0

        for graph in train_graphs:
            x = graph["x"].to(device)
            edge_index = graph["edge_index"].to(device)
            edge_attr = graph["edge_attr"].to(device)
            y = graph["y"].to(device)

            # Normalize targets
            normalized_y = (y - target_mean) / target_std

            prediction, _ = model(x, edge_index, edge_attr)
            loss = loss_function(prediction, normalized_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(y)
            total_nodes += len(y)

        validation_mae = evaluate(
            model, validation_graphs, target_mean, target_std, device
        )

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 5 == 0:
            training_loss = total_loss / max(total_nodes, 1)
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {training_loss:.6f} | "
                f"Validation MAE: {validation_mae:.6f}"
            )

    # Restore best validation model
    if best_state is not None:
        model.load_state_dict(best_state)

    test_mae = evaluate(model, test_graphs, target_mean, target_std, device)
    print(f"Test MAE: {test_mae:.6f}")

    metadata = {
        "device": str(device),
        "best_validation_mae": best_validation_mae,
        "test_mae": test_mae,
        "target_mean": target_mean.detach().cpu(),
        "target_std": target_std.detach().cpu(),
    }

    return model, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    graphs = torch.load(args.graphs, map_location="cpu", weights_only=False)

    model, metadata = train(graphs, epochs=args.epochs, learning_rate=args.lr)

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint,
    )

    print(f"Checkpoint saved to: {checkpoint}")


if __name__ == "__main__":
    main()