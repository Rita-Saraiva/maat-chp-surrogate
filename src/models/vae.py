from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import DataLoader

from src.config.schema import TrainLoopConfig, VAEProfileConfig
from src.eval.metrics import evaluate_regression
from src.models.base import SurrogateModel
from src.train.loop import fit_torch_model, predict_torch_model, select_device
from src.train.transform import TensorStandardizer


def beta_for_epoch(
    epoch: int,
    beta_start: float,
    beta_final: float,
    warmup_epochs: int,
) -> float:
    if warmup_epochs <= 0 or epoch >= warmup_epochs:
        return float(beta_final)
    fraction = max(0, epoch) / float(warmup_epochs)
    return float(beta_start) + (float(beta_final) - float(beta_start)) * fraction


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class _MLP(torch.nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        widths = [int(value) for value in dims]
        layers: list[torch.nn.Module] = []
        for index in range(len(widths) - 1):
            layers.append(torch.nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                layers.append(torch.nn.GELU())
                if dropout > 0.0:
                    layers.append(torch.nn.Dropout(dropout))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class VariationalNetwork(torch.nn.Module):
    """q(z|x) -> decode(z) -> y. Trains with reparameterization, evaluates with mu."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_hidden_dims: list[int],
        latent_dim: int,
        dropout: float,
        decoder_hidden_dims: list[int] | None,
    ) -> None:
        super().__init__()
        encoder_hidden = [int(value) for value in encoder_hidden_dims]
        decoder_hidden = [
            int(value)
            for value in (decoder_hidden_dims or list(reversed(encoder_hidden)))
        ]
        latent = int(latent_dim)

        self.backbone = _MLP([int(input_dim), *encoder_hidden], dropout)
        self.mu_head = torch.nn.Linear(encoder_hidden[-1], latent)
        self.logvar_head = torch.nn.Linear(encoder_hidden[-1], latent)
        self.decoder = _MLP([latent, *decoder_hidden, int(output_dim)], dropout)

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
        return self.decoder(latent)


class VAESurrogate(SurrogateModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        profile: VAEProfileConfig,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.profile = profile
        self.network = VariationalNetwork(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            encoder_hidden_dims=profile.encoder_hidden_dims,
            latent_dim=profile.latent_dim,
            dropout=profile.dropout,
            decoder_hidden_dims=profile.decoder_hidden_dims,
        )
        self.standardizer: TensorStandardizer | None = None
        self.device = "cpu"

    def _auxiliary_loss(self, model: torch.nn.Module, epoch: int) -> torch.Tensor:
        network = model
        if network.last_mu is None or network.last_logvar is None:
            raise RuntimeError("VAE forward must run before computing the KL term")
        beta = beta_for_epoch(
            epoch=epoch,
            beta_start=self.profile.beta_start,
            beta_final=self.profile.beta_final,
            warmup_epochs=self.profile.beta_warmup_epochs,
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
        training_result = fit_torch_model(
            model=self.network,
            train_loader=train_loader,
            validation_loader=validation_loader,
            standardizer=self.standardizer,
            config=config,
            checkpoint_path=checkpoint_path,
            auxiliary_loss=self._auxiliary_loss,
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
                "family": "VAE",
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
