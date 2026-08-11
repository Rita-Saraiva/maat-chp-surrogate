from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from src.models.base import SurrogateModel


def evaluate_surrogate(
    model: SurrogateModel,
    validation_loader: DataLoader,
    test_loader: DataLoader,
) -> dict[str, dict[str, Any]]:
    return {
        "Validation": model.evaluate(validation_loader),
        "Test": model.evaluate(test_loader),
    }
