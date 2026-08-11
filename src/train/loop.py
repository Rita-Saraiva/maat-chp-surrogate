from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.schema import TrainLoopConfig
from src.train.transform import TensorStandardizer


class EarlyStopping:
    def __init__(self, patience: int, minimum_delta: float) -> None:
        self.patience = int(patience)
        self.minimum_delta = float(minimum_delta)
        self.best_value: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best_value is None or value < self.best_value - self.minimum_delta:
            self.best_value = float(value)
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def set_random_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_loss(name: str) -> torch.nn.Module:
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "mae":
        return torch.nn.L1Loss()
    if name == "huber":
        return torch.nn.HuberLoss(delta=1.0)
    raise ValueError(f"Unsupported loss {name!r}")


def create_optimizer(model: torch.nn.Module, config: TrainLoopConfig) -> torch.optim.Optimizer:
    parameters = model.parameters()
    learning_rate = config.optimizer.learning_rate
    weight_decay = config.optimizer.weight_decay
    if config.optimizer.name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if config.optimizer.name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer {config.optimizer.name!r}")


def compute_epoch_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    standardizer: TensorStandardizer,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float | None,
    epoch: int = 0,
    auxiliary_loss: Callable[[torch.nn.Module, int], torch.Tensor] | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_rows = 0

    for x_batch, y_batch in loader:
        x_values = x_batch.to(device, non_blocking=True)
        y_values = y_batch.to(device, non_blocking=True)
        x_scaled = standardizer.transform_inputs(x_values)
        y_scaled = standardizer.transform_targets(y_values)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        predictions = model(x_scaled)
        loss = criterion(predictions, y_scaled)
        if training and auxiliary_loss is not None:
            loss = loss + auxiliary_loss(model, epoch)

        if optimizer is not None:
            loss.backward()
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        batch_rows = int(x_values.shape[0])
        total_loss += loss.detach().to(torch.float64) * batch_rows
        total_rows += batch_rows

    if total_rows == 0:
        raise ValueError("Cannot calculate epoch loss for an empty loader")
    return float((total_loss / total_rows).item())


def fit_torch_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    standardizer: TensorStandardizer,
    config: TrainLoopConfig,
    checkpoint_path: str | Path,
    auxiliary_loss: Callable[[torch.nn.Module, int], torch.Tensor] | None = None,
    periodic_save_callback: Callable[[int], None] | None = None,
    periodic_save_every: int = 25,
) -> dict[str, Any]:
    set_random_seed(config.seed, config.deterministic)
    device = select_device(config.device)
    model.to(device)
    standardizer.move(device)

    criterion = create_loss(config.loss)
    optimizer = create_optimizer(model, config)
    scheduler = None
    if config.scheduler.enabled:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=config.scheduler.minimum_learning_rate,
        )

    stopper = EarlyStopping(config.patience, config.minimum_delta)
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        train_loss = compute_epoch_loss(
            model=model,
            loader=train_loader,
            criterion=criterion,
            standardizer=standardizer,
            device=device,
            optimizer=optimizer,
            gradient_clip=config.gradient_clip,
            epoch=epoch,
            auxiliary_loss=auxiliary_loss,
        )
        validation_loss = compute_epoch_loss(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            standardizer=standardizer,
            device=device,
            optimizer=None,
            gradient_clip=None,
        )
        if scheduler is not None:
            scheduler.step(validation_loss)

        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Validation Loss": validation_loss,
                "Learning Rate": learning_rate,
            }
        )

        if epoch == 1 or epoch % config.print_every == 0 or epoch == config.epochs:
            print(
                f"[training] epoch={epoch}/{config.epochs} "
                f"train_loss={train_loss:.8f} "
                f"validation_loss={validation_loss:.8f} "
                f"learning_rate={learning_rate:.3e}",
                flush=True,
            )

        if validation_loss < best_validation_loss:
            best_validation_loss = float(validation_loss)
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        if periodic_save_callback is not None and epoch % periodic_save_every == 0:
            periodic_save_callback(epoch)

        if stopper.step(validation_loss):
            print(f"[training] early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint")

    model.load_state_dict(best_state)
    model.to(device)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, checkpoint)

    return {
        "Device": device,
        "Best Epoch": best_epoch,
        "Best Validation Loss": best_validation_loss,
        "Epochs Trained": len(history),
        "History": copy.deepcopy(history),
    }


def predict_torch_model(
    model: torch.nn.Module,
    loader: DataLoader,
    standardizer: TensorStandardizer,
    device: str,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x_values = x_batch.to(device, non_blocking=True)
            x_scaled = standardizer.transform_inputs(x_values)
            prediction_scaled = model(x_scaled)
            prediction = standardizer.inverse_targets(prediction_scaled)
            predictions.append(prediction.detach().cpu())
            targets.append(y_batch.detach().cpu())

    if not predictions:
        raise ValueError("Prediction loader produced no batches")
    return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)
