"""Multi-head surrogates: one shared trunk, one output head per component.

Every head reproduces the exact single-head output for its component (that
component's columns flattened region-major), and the heads are concatenated into
one tensor. Training uses the ordinary flat loss over the concatenated output, so
the shared training loop, standardizer, and predictor are reused unchanged. VAE
keeps its KL auxiliary term.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import DataLoader

from src.config.schema import (
    AutoProfileConfig,
    MLPProfileConfig,
    TrainLoopConfig,
    VAEProfileConfig,
)
from src.eval.metrics import evaluate_regression_multihead
from src.models.auto import GELUBlock
from src.models.base import SurrogateModel
from src.models.mlp import create_activation
from src.models.vae import _MLP, beta_for_epoch, kl_divergence
from src.train.loop import fit_torch_model, predict_torch_model, select_device
from src.train.transform import TensorStandardizer


def _ordered_heads(head_output_dims: Mapping[str, int]) -> list[tuple[str, int]]:
    heads = [(str(name), int(dim)) for name, dim in head_output_dims.items()]
    if not heads:
        raise ValueError("head_output_dims must contain at least one head")
    if any(dim <= 0 for _, dim in heads):
        raise ValueError("All head output dimensions must be positive")
    return heads


class MultiHeadMLPNetwork(torch.nn.Module):
    """Shared MLP trunk -> one linear head per component -> concatenated output."""

    def __init__(
        self,
        input_dim: int,
        head_output_dims: Mapping[str, int],
        profile: MLPProfileConfig,
    ) -> None:
        super().__init__()
        heads = _ordered_heads(head_output_dims)
        hidden = [int(value) for value in profile.hidden_dims]
        dimensions = [int(input_dim), *hidden]

        layers: list[torch.nn.Module] = []
        for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:]):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if profile.layer_norm:
                layers.append(torch.nn.LayerNorm(out_dim))
            layers.append(create_activation(profile.activation))
            if profile.dropout > 0.0:
                layers.append(torch.nn.Dropout(profile.dropout))
        self.trunk = torch.nn.Sequential(*layers)

        trunk_out = hidden[-1]
        self.head_names = [name for name, _ in heads]
        self.heads = torch.nn.ModuleDict(
            {name: torch.nn.Linear(trunk_out, dim) for name, dim in heads}
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(values)
        return torch.cat([self.heads[name](shared) for name in self.head_names], dim=1)


class MultiHeadBottleneckNetwork(torch.nn.Module):
    """Auto encoder -> latent -> predictor trunk -> one head per component."""

    def __init__(
        self,
        input_dim: int,
        head_output_dims: Mapping[str, int],
        profile: AutoProfileConfig,
    ) -> None:
        super().__init__()
        heads = _ordered_heads(head_output_dims)
        hidden = [int(value) for value in profile.hidden_dims]
        latent = int(profile.latent_dim)
        dropout = profile.dropout

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
        self.predictor = torch.nn.Sequential(*predictor)

        trunk_out = hidden[0]
        self.head_names = [name for name, _ in heads]
        self.heads = torch.nn.ModuleDict(
            {name: torch.nn.Linear(trunk_out, dim) for name, dim in heads}
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        shared = self.predictor(self.encoder(values))
        return torch.cat([self.heads[name](shared) for name in self.head_names], dim=1)


class MultiHeadVariationalNetwork(torch.nn.Module):
    """VAE q(z|x) -> decoder trunk -> one head per component. Keeps mu/logvar."""

    def __init__(
        self,
        input_dim: int,
        head_output_dims: Mapping[str, int],
        profile: VAEProfileConfig,
    ) -> None:
        super().__init__()
        heads = _ordered_heads(head_output_dims)
        encoder_hidden = [int(value) for value in profile.encoder_hidden_dims]
        decoder_hidden = [
            int(value)
            for value in (profile.decoder_hidden_dims or list(reversed(encoder_hidden)))
        ]
        latent = int(profile.latent_dim)
        dropout = profile.dropout

        self.backbone = _MLP([int(input_dim), *encoder_hidden], dropout)
        self.mu_head = torch.nn.Linear(encoder_hidden[-1], latent)
        self.logvar_head = torch.nn.Linear(encoder_hidden[-1], latent)

        decoder_layers: list[torch.nn.Module] = []
        previous = latent
        for width in decoder_hidden:
            decoder_layers.append(torch.nn.Linear(previous, width))
            decoder_layers.append(torch.nn.GELU())
            if dropout > 0.0:
                decoder_layers.append(torch.nn.Dropout(dropout))
            previous = width
        self.decoder_trunk = torch.nn.Sequential(*decoder_layers)

        trunk_out = decoder_hidden[-1]
        self.head_names = [name for name, _ in heads]
        self.heads = torch.nn.ModuleDict(
            {name: torch.nn.Linear(trunk_out, dim) for name, dim in heads}
        )

        self.last_mu: torch.Tensor | None = None
        self.last_logvar: torch.Tensor | None = None

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(values)
        mu = self.mu_head(hidden)
        logvar = torch.clamp(self.logvar_head(hidden), min=-20.0, max=10.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(values)
        latent = self.reparameterize(mu, logvar) if self.training else mu
        self.last_mu = mu
        self.last_logvar = logvar
        shared = self.decoder_trunk(latent)
        return torch.cat([self.heads[name](shared) for name in self.head_names], dim=1)


def build_multihead_network(
    family: str,
    input_dim: int,
    head_output_dims: Mapping[str, int],
    profile: MLPProfileConfig | AutoProfileConfig | VAEProfileConfig,
) -> torch.nn.Module:
    if family == "MLP":
        return MultiHeadMLPNetwork(input_dim, head_output_dims, profile)  # type: ignore[arg-type]
    if family == "Auto":
        return MultiHeadBottleneckNetwork(input_dim, head_output_dims, profile)  # type: ignore[arg-type]
    if family == "VAE":
        return MultiHeadVariationalNetwork(input_dim, head_output_dims, profile)  # type: ignore[arg-type]
    raise ValueError(f"Multi-head training does not support family {family!r}")


class MultiHeadSurrogate(SurrogateModel):
    """Shared-trunk multi-head regressor with flat loss over concatenated heads."""

    def __init__(
        self,
        family: str,
        input_dim: int,
        head_output_dims: Mapping[str, int],
        head_slices: Mapping[str, tuple[int, int]],
        profile: MLPProfileConfig | AutoProfileConfig | VAEProfileConfig,
    ) -> None:
        self.family = str(family)
        self.input_dim = int(input_dim)
        self.head_output_dims = {str(name): int(dim) for name, dim in head_output_dims.items()}
        self.head_slices = {
            str(name): (int(bounds[0]), int(bounds[1]))
            for name, bounds in head_slices.items()
        }
        self.target_names = list(self.head_output_dims.keys())
        self.output_dim = sum(self.head_output_dims.values())
        self.profile = profile
        self.network = build_multihead_network(
            self.family, self.input_dim, self.head_output_dims, profile
        )
        self.standardizer: TensorStandardizer | None = None
        self.device = "cpu"

    def _auxiliary_loss(self, model: torch.nn.Module, epoch: int) -> torch.Tensor:
        if self.family != "VAE":
            raise RuntimeError("Auxiliary loss is only defined for the VAE family")
        network = model
        if network.last_mu is None or network.last_logvar is None:
            raise RuntimeError("VAE forward must run before computing the KL term")
        profile = self.profile
        assert isinstance(profile, VAEProfileConfig)
        beta = beta_for_epoch(
            epoch=epoch,
            beta_start=profile.beta_start,
            beta_final=profile.beta_final,
            warmup_epochs=profile.beta_warmup_epochs,
        )
        return beta * kl_divergence(network.last_mu, network.last_logvar)

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
        auxiliary_loss = self._auxiliary_loss if self.family == "VAE" else None
        training_result = fit_torch_model(
            model=self.network,
            train_loader=train_loader,
            validation_loader=validation_loader,
            standardizer=self.standardizer,
            config=config,
            checkpoint_path=checkpoint_path,
            auxiliary_loss=auxiliary_loss,
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
        return evaluate_regression_multihead(
            model=self.network,
            loader=loader,
            device=self.device,
            standardizer=self.standardizer,
            head_slices=self.head_slices,
        )

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        if self.standardizer is None:
            raise RuntimeError("The model must be fitted or loaded before saving")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "family": self.family,
                "mode": "multi",
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "profile": self.profile.model_dump(mode="python"),
                "target_names": list(self.target_names),
                "head_output_dims": dict(self.head_output_dims),
                "head_slices": {name: list(bounds) for name, bounds in self.head_slices.items()},
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
        checkpoint_heads = {str(name): int(dim) for name, dim in checkpoint["head_output_dims"].items()}
        if checkpoint_heads != self.head_output_dims:
            raise ValueError(
                f"Checkpoint heads={checkpoint_heads} do not match model heads={self.head_output_dims}"
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
