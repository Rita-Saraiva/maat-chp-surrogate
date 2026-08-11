from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from src.config.schema import AutoProfileConfig, TrainLoopConfig
from src.eval.metrics import evaluate_regression
from src.models.base import SurrogateModel
from src.train.loop import fit_torch_model, predict_torch_model, select_device
from src.train.transform import TensorStandardizer


class GELUBlock(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class BottleneckNetwork(torch.nn.Module):
    """Supervised bottleneck: x -> encoder -> latent -> predictor -> y."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden = [int(value) for value in hidden_dims]
        latent = int(latent_dim)

        encoder: list[torch.nn.Module] = []
        previous = int(input_dim)
        for width in hidden:
            encoder.append(GELUBlock(previous, width, dropout))
            previous = width
        encoder.append(torch.nn.Linear(previous, latent))
        self.encoder = torch.nn.Sequential(*encoder)

        predictor: list[torch.nn.Module] = []
        previous = latent
        for width in reversed(hidden):
            predictor.append(GELUBlock(previous, width, dropout))
            previous = width
        predictor.append(torch.nn.Linear(previous, int(output_dim)))
        self.predictor = torch.nn.Sequential(*predictor)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.encoder(values))


class AutoSurrogate(SurrogateModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        profile: AutoProfileConfig,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.profile = profile
        self.network = BottleneckNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=profile.hidden_dims,
            latent_dim=profile.latent_dim,
            dropout=profile.dropout,
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
                "family": "Auto",
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
