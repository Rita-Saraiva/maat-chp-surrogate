from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from src.config.schema import MLPProfileConfig, TrainLoopConfig
from src.eval.metrics import evaluate_regression
from src.models.base import SurrogateModel
from src.train.loop import fit_torch_model, predict_torch_model, select_device
from src.train.transform import TensorStandardizer


class MLPNetwork(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float,
        activation: str,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        dimensions = [int(input_dim), *[int(value) for value in hidden_dims]]
        layers: list[torch.nn.Module] = []

        for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(torch.nn.LayerNorm(out_dim))
            layers.append(create_activation(activation))
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))

        layers.append(torch.nn.Linear(dimensions[-1], int(output_dim)))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def create_activation(name: str) -> torch.nn.Module:
    if name == "gelu":
        return torch.nn.GELU()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "silu":
        return torch.nn.SiLU()
    raise ValueError(f"Unsupported activation {name!r}")


class MLPSurrogate(SurrogateModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        profile: MLPProfileConfig,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.profile = profile
        self.network = MLPNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=profile.hidden_dims,
            dropout=profile.dropout,
            activation=profile.activation,
            layer_norm=profile.layer_norm,
        )
        self.standardizer: TensorStandardizer | None = None
        self.device = "cpu"

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        config: TrainLoopConfig,
        checkpoint_path: str | Path,
        periodic_save_callback: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        self.standardizer = TensorStandardizer(
            standardize_inputs=config.standardization.inputs,
            standardize_targets=config.standardization.targets,
            epsilon=config.standardization.epsilon,
        )
        print("[training] fitting standardization statistics", flush=True)
        training_dataset = getattr(train_loader, "dataset", None)
        materialized = (
            training_dataset.materialized_tensors()
            if hasattr(training_dataset, "materialized_tensors")
            else None
        )
        if materialized is not None:
            self.standardizer.fit_from_tensors(materialized[0], materialized[1])
        else:
            self.standardizer.fit(train_loader)
        training_result = fit_torch_model(
            model=self.network,
            train_loader=train_loader,
            validation_loader=validation_loader,
            standardizer=self.standardizer,
            config=config,
            checkpoint_path=checkpoint_path,
            periodic_save_callback=periodic_save_callback,
        )
        self.device = str(training_result["Device"])
        return training_result

    def predict(
        self,
        loader: DataLoader,
        max_batches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.standardizer is None:
            raise RuntimeError("The model must be fitted or loaded before prediction")
        self.network.to(self.device)
        self.standardizer.move(self.device)
        return predict_torch_model(
            model=self.network,
            loader=loader,
            standardizer=self.standardizer,
            device=self.device,
            max_batches=max_batches,
        )

    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        if self.standardizer is None:
            raise RuntimeError("The model must be fitted or loaded before evaluation")
        self.network.to(self.device)
        self.standardizer.move(self.device)
        return evaluate_regression(
            model=self.network,
            loader=loader,
            device=self.device,
            standardizer=self.standardizer,
        )

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        if self.standardizer is None:
            raise RuntimeError("The model must be fitted or loaded before saving")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "family": "MLP",
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "profile": self.profile.model_dump(mode="python"),
                "model_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.network.state_dict().items()
                },
                "standardizer_state": self.standardizer.state_dict(),
                "metadata": metadata or {},
            },
            output_path,
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        if int(checkpoint["input_dim"]) != self.input_dim:
            raise ValueError(
                f"Checkpoint input_dim={checkpoint['input_dim']} does not match model input_dim={self.input_dim}"
            )
        if int(checkpoint["output_dim"]) != self.output_dim:
            raise ValueError(
                f"Checkpoint output_dim={checkpoint['output_dim']} does not match model output_dim={self.output_dim}"
            )
        self.network.load_state_dict(checkpoint["model_state"])
        standardizer_state = checkpoint["standardizer_state"]
        self.standardizer = TensorStandardizer(
            standardize_inputs=bool(standardizer_state["standardize_inputs"]),
            standardize_targets=bool(standardizer_state["standardize_targets"]),
            epsilon=float(standardizer_state["epsilon"]),
        )
        self.standardizer.load_state_dict(standardizer_state)
        self.device = select_device("auto")
        return dict(checkpoint.get("metadata", {}))
