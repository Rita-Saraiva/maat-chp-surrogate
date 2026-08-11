from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


class TensorStandardizer:
    def __init__(self, standardize_inputs: bool, standardize_targets: bool, epsilon: float) -> None:
        self.standardize_inputs = bool(standardize_inputs)
        self.standardize_targets = bool(standardize_targets)
        self.epsilon = float(epsilon)
        self.x_mean: torch.Tensor | None = None
        self.x_scale: torch.Tensor | None = None
        self.y_mean: torch.Tensor | None = None
        self.y_scale: torch.Tensor | None = None

    def fit(self, loader: DataLoader) -> None:
        row_count = 0
        x_sum: torch.Tensor | None = None
        x_square_sum: torch.Tensor | None = None
        y_sum: torch.Tensor | None = None
        y_square_sum: torch.Tensor | None = None

        for x_batch, y_batch in loader:
            batch_rows = int(x_batch.shape[0])
            row_count += batch_rows

            current_x_sum = x_batch.sum(dim=0, dtype=torch.float64)
            current_x_square_sum = x_batch.square().sum(dim=0, dtype=torch.float64)
            current_y_sum = y_batch.sum(dim=0, dtype=torch.float64)
            current_y_square_sum = y_batch.square().sum(dim=0, dtype=torch.float64)

            x_sum = current_x_sum if x_sum is None else x_sum + current_x_sum
            x_square_sum = (
                current_x_square_sum
                if x_square_sum is None
                else x_square_sum + current_x_square_sum
            )
            y_sum = current_y_sum if y_sum is None else y_sum + current_y_sum
            y_square_sum = (
                current_y_square_sum
                if y_square_sum is None
                else y_square_sum + current_y_square_sum
            )

        if row_count == 0 or x_sum is None or y_sum is None:
            raise ValueError("Cannot fit standardization on an empty loader")

        self.finalize(x_sum, x_square_sum, y_sum, y_square_sum, row_count)

    def fit_from_tensors(self, x_values: torch.Tensor, y_values: torch.Tensor) -> None:
        """Fit statistics in a single vectorized pass over in-RAM tensors."""
        row_count = int(x_values.shape[0])
        if row_count == 0:
            raise ValueError("Cannot fit standardization on empty tensors")
        x_sum = x_values.sum(dim=0, dtype=torch.float64)
        x_square_sum = x_values.square().sum(dim=0, dtype=torch.float64)
        y_sum = y_values.sum(dim=0, dtype=torch.float64)
        y_square_sum = y_values.square().sum(dim=0, dtype=torch.float64)
        self.finalize(x_sum, x_square_sum, y_sum, y_square_sum, row_count)

    def finalize(
        self,
        x_sum: torch.Tensor,
        x_square_sum: torch.Tensor,
        y_sum: torch.Tensor,
        y_square_sum: torch.Tensor,
        row_count: int,
    ) -> None:
        x_mean = x_sum / row_count
        y_mean = y_sum / row_count
        x_variance = torch.clamp(x_square_sum / row_count - x_mean.square(), min=0.0)
        y_variance = torch.clamp(y_square_sum / row_count - y_mean.square(), min=0.0)

        self.x_mean = x_mean.to(dtype=torch.float32)
        self.y_mean = y_mean.to(dtype=torch.float32)
        self.x_scale = torch.sqrt(x_variance).to(dtype=torch.float32).clamp_min(self.epsilon)
        self.y_scale = torch.sqrt(y_variance).to(dtype=torch.float32).clamp_min(self.epsilon)

        if not self.standardize_inputs:
            self.x_mean.zero_()
            self.x_scale.fill_(1.0)
        if not self.standardize_targets:
            self.y_mean.zero_()
            self.y_scale.fill_(1.0)

    def ensure_fitted(self) -> None:
        values = (self.x_mean, self.x_scale, self.y_mean, self.y_scale)
        if any(value is None for value in values):
            raise RuntimeError("TensorStandardizer has not been fitted")

    def move(self, device: str | torch.device) -> None:
        self.ensure_fitted()
        self.x_mean = self.x_mean.to(device)
        self.x_scale = self.x_scale.to(device)
        self.y_mean = self.y_mean.to(device)
        self.y_scale = self.y_scale.to(device)

    def transform_inputs(self, values: torch.Tensor) -> torch.Tensor:
        self.ensure_fitted()
        return (values - self.x_mean) / self.x_scale

    def transform_targets(self, values: torch.Tensor) -> torch.Tensor:
        self.ensure_fitted()
        return (values - self.y_mean) / self.y_scale

    def inverse_targets(self, values: torch.Tensor) -> torch.Tensor:
        self.ensure_fitted()
        return values * self.y_scale + self.y_mean

    def state_dict(self) -> dict[str, Any]:
        self.ensure_fitted()
        return {
            "standardize_inputs": self.standardize_inputs,
            "standardize_targets": self.standardize_targets,
            "epsilon": self.epsilon,
            "x_mean": self.x_mean.detach().cpu(),
            "x_scale": self.x_scale.detach().cpu(),
            "y_mean": self.y_mean.detach().cpu(),
            "y_scale": self.y_scale.detach().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.standardize_inputs = bool(state["standardize_inputs"])
        self.standardize_targets = bool(state["standardize_targets"])
        self.epsilon = float(state["epsilon"])
        self.x_mean = torch.as_tensor(state["x_mean"], dtype=torch.float32)
        self.x_scale = torch.as_tensor(state["x_scale"], dtype=torch.float32)
        self.y_mean = torch.as_tensor(state["y_mean"], dtype=torch.float32)
        self.y_scale = torch.as_tensor(state["y_scale"], dtype=torch.float32)
