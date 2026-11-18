from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class GNNPrediction:
    """
    Behavior prediction for one ego vehicle.

    Values are in physical units:
        speed       : m/s
        lon_acc     : m/s^2
        lat_acc     : m/s^2
    """

    speed: float
    lon_acc: float
    lat_acc: float

    raw_speed: float
    raw_lon_acc: float
    raw_lat_acc: float

    normalized_speed: float
    normalized_lon_acc: float
    normalized_lat_acc: float

    clipped: bool


class GNNSafePredictor:
    """
    Loads the trained InteractionGNN and provides a safe inference interface.

    The model itself is NOT modified.

    The prediction pipeline is:

        graph
          -> normalized model output
          -> denormalization
          -> physical sanity constraints
          -> GNNPrediction
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_class,
        device: str | torch.device = "cpu",
        max_speed: float = 40.0,
        max_longitudinal_acc: float = 10.0,
        max_lateral_acc: float = 10.0,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)

        self.max_speed = float(max_speed)
        self.max_longitudinal_acc = float(max_longitudinal_acc)
        self.max_lateral_acc = float(max_lateral_acc)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.target_mean = np.asarray(
            checkpoint["target_mean"],
            dtype=np.float32,
        )

        self.target_std = np.asarray(
            checkpoint["target_std"],
            dtype=np.float32,
        )

        self.model = model_class(
            node_dim=int(checkpoint["node_dim"]),
            edge_dim=int(checkpoint["edge_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            out_dim=int(checkpoint["out_dim"]),
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        graph: dict[str, Any],
        ego_index: int,
    ) -> GNNPrediction:
        """
        Predict behavior for one ego node.

        The graph is expected to contain:
            x
            edge_index
            edge_attr
        """

        x = graph["x"]

        edge_index = graph["edge_index"]
        edge_attr = graph["edge_attr"]

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)

        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.as_tensor(
                edge_index,
                dtype=torch.long,
            )

        if not isinstance(edge_attr, torch.Tensor):
            edge_attr = torch.as_tensor(
                edge_attr,
                dtype=torch.float32,
            )

        x = x.to(self.device, dtype=torch.float32)
        edge_index = edge_index.to(self.device, dtype=torch.long)
        edge_attr = edge_attr.to(self.device, dtype=torch.float32)

        prediction_normalized, _ = self.model(
            x,
            edge_index,
            edge_attr,
        )

        if ego_index < 0 or ego_index >= prediction_normalized.shape[0]:
            raise IndexError(
                f"ego_index={ego_index} outside graph with "
                f"{prediction_normalized.shape[0]} nodes"
            )

        normalized = (
            prediction_normalized[ego_index]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        if normalized.shape[0] != 3:
            raise ValueError(
                f"Expected 3 prediction values, got {normalized.shape[0]}"
            )

        raw = (
            normalized * self.target_std.astype(np.float64)
            + self.target_mean.astype(np.float64)
        )

        raw_speed = float(raw[0])
        raw_lon_acc = float(raw[1])
        raw_lat_acc = float(raw[2])

        speed = float(
            np.clip(
                raw_speed,
                0.0,
                self.max_speed,
            )
        )

        lon_acc = float(
            np.clip(
                raw_lon_acc,
                -self.max_longitudinal_acc,
                self.max_longitudinal_acc,
            )
        )

        lat_acc = float(
            np.clip(
                raw_lat_acc,
                -self.max_lateral_acc,
                self.max_lateral_acc,
            )
        )

        clipped = not np.allclose(
            [raw_speed, raw_lon_acc, raw_lat_acc],
            [speed, lon_acc, lat_acc],
            rtol=0.0,
            atol=1e-12,
        )

        return GNNPrediction(
            speed=speed,
            lon_acc=lon_acc,
            lat_acc=lat_acc,
            raw_speed=raw_speed,
            raw_lon_acc=raw_lon_acc,
            raw_lat_acc=raw_lat_acc,
            normalized_speed=float(normalized[0]),
            normalized_lon_acc=float(normalized[1]),
            normalized_lat_acc=float(normalized[2]),
            clipped=clipped,
        )


def prediction_to_dict(prediction: GNNPrediction) -> dict[str, float | bool]:
    return {
        "speed": prediction.speed,
        "lon_acc": prediction.lon_acc,
        "lat_acc": prediction.lat_acc,
        "raw_speed": prediction.raw_speed,
        "raw_lon_acc": prediction.raw_lon_acc,
        "raw_lat_acc": prediction.raw_lat_acc,
        "normalized_speed": prediction.normalized_speed,
        "normalized_lon_acc": prediction.normalized_lon_acc,
        "normalized_lat_acc": prediction.normalized_lat_acc,
        "clipped": prediction.clipped,
    }