from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from src.config.schema import TrainLoopConfig


class SurrogateModel(ABC):
    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        config: TrainLoopConfig,
        checkpoint_path: str | Path,
        periodic_save_callback: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        loader: DataLoader,
        max_batches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path) -> dict[str, Any]:
        raise NotImplementedError
