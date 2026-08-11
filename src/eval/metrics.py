from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.train.transform import TensorStandardizer


# Policy rank metrics are correlation based, so unlike the streaming regression
# accumulators they need the full prediction/target arrays in memory. To bound the
# cost of Kendall's tau (which is ``O(n log n)`` and can blow up on 800+ output
# dimensions), the flattened variant is deterministically subsampled to at most
# ``POLICY_MAX_POINTS`` values using ``POLICY_SUBSAMPLE_SEED``.
POLICY_MAX_POINTS = 500_000
POLICY_SUBSAMPLE_SEED = 42


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator != 0.0 else float("nan")


class _RegressionAccumulator:
    """Streaming sums for regression metrics over a (possibly sliced) tensor."""

    def __init__(self) -> None:
        self.count = 0
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.target_sum = 0.0
        self.target_square_sum = 0.0
        self.maximum_absolute_error = 0.0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        errors = predictions - targets
        absolute_errors = errors.abs()
        self.count += int(targets.numel())
        self.absolute_error_sum += float(absolute_errors.sum().item())
        self.squared_error_sum += float(errors.square().sum().item())
        self.target_sum += float(targets.sum().item())
        self.target_square_sum += float(targets.square().sum().item())
        if targets.numel() > 0:
            self.maximum_absolute_error = max(
                self.maximum_absolute_error, float(absolute_errors.max().item())
            )

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            raise ValueError("Cannot evaluate an empty loader")
        mean_target = self.target_sum / self.count
        total_square = self.target_square_sum - self.count * mean_target * mean_target
        r2 = 1.0 - self.squared_error_sum / total_square if total_square > 0.0 else float("nan")
        return {
            "MSE": self.squared_error_sum / self.count,
            "RMSE": math.sqrt(self.squared_error_sum / self.count),
            "MAE": self.absolute_error_sum / self.count,
            "R2": r2,
            "Max Absolute Error": self.maximum_absolute_error,
            "Values": self.count,
        }


def evaluate_regression(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    standardizer: TensorStandardizer,
) -> dict[str, Any]:
    model.eval()
    count = 0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    target_sum = 0.0
    target_square_sum = 0.0
    maximum_absolute_error = 0.0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_values = x_batch.to(device, non_blocking=True)
            y_values = y_batch.to(device, non_blocking=True)
            x_scaled = standardizer.transform_inputs(x_values)
            prediction_scaled = model(x_scaled)
            predictions = standardizer.inverse_targets(prediction_scaled)

            errors = predictions - y_values
            absolute_errors = errors.abs()
            squared_errors = errors.square()

            count += int(y_values.numel())
            absolute_error_sum += float(absolute_errors.sum().item())
            squared_error_sum += float(squared_errors.sum().item())
            target_sum += float(y_values.sum().item())
            target_square_sum += float(y_values.square().sum().item())
            maximum_absolute_error = max(
                maximum_absolute_error,
                float(absolute_errors.max().item()),
            )

    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")

    mean_target = target_sum / count
    total_square = target_square_sum - count * mean_target * mean_target
    r2 = 1.0 - squared_error_sum / total_square if total_square > 0.0 else float("nan")

    return {
        "MSE": squared_error_sum / count,
        "RMSE": math.sqrt(squared_error_sum / count),
        "MAE": absolute_error_sum / count,
        "R2": r2,
        "Max Absolute Error": maximum_absolute_error,
        "Values": count,
    }


def evaluate_regression_multihead(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    standardizer: TensorStandardizer,
    head_slices: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    """Overall and per-head regression metrics for a concatenated multi-head output.

    Predictions are evaluated in the original (inverse-standardized) target space,
    identically to :func:`evaluate_regression`. Each head is scored on its
    contiguous slice of the concatenated output, matching its single-head run.
    """
    model.eval()
    overall = _RegressionAccumulator()
    per_head = {name: _RegressionAccumulator() for name in head_slices}

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_values = x_batch.to(device, non_blocking=True)
            y_values = y_batch.to(device, non_blocking=True)
            x_scaled = standardizer.transform_inputs(x_values)
            prediction_scaled = model(x_scaled)
            predictions = standardizer.inverse_targets(prediction_scaled)

            overall.update(predictions, y_values)
            for name, (start, stop) in head_slices.items():
                per_head[name].update(predictions[:, start:stop], y_values[:, start:stop])

    return {
        "Overall": overall.result(),
        "Per Head": {name: accumulator.result() for name, accumulator in per_head.items()},
    }


def multihead_metrics_from_arrays(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    head_slices: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    """Overall and per-head regression metrics from materialized prediction arrays.

    This mirrors :func:`evaluate_regression_multihead` but consumes already
    computed ``(targets, predictions)`` tensors in the original target space. It is
    used by families such as GP whose prediction path returns concatenated arrays
    rather than a plain ``torch.nn.Module`` callable.
    """
    overall = _RegressionAccumulator()
    overall.update(predictions, targets)
    per_head: dict[str, Any] = {}
    for name, (start, stop) in head_slices.items():
        accumulator = _RegressionAccumulator()
        accumulator.update(predictions[:, start:stop], targets[:, start:stop])
        per_head[name] = accumulator.result()

    return {"Overall": overall.result(), "Per Head": per_head}


def _to_numpy(values: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _rank_correlations(true_arr: np.ndarray, pred_arr: np.ndarray) -> dict[str, Any]:
    """Kendall's tau and Spearman's rho between two 1-D arrays.

    ``scipy`` is imported lazily so environments without it (e.g. a container that
    ships without ``scipy``) degrade gracefully to ``nan`` values plus a note,
    exactly like the other optional-dependency paths in this project.
    """
    kendall = float("nan")
    spearman = float("nan")
    note = "ok"
    if true_arr.size == 0 or pred_arr.size == 0:
        note = "empty_inputs"
    elif true_arr.size != pred_arr.size:
        note = "size_mismatch"
    else:
        try:
            from scipy.stats import kendalltau, spearmanr

            kendall_value, _ = kendalltau(true_arr, pred_arr)
            spearman_value, _ = spearmanr(true_arr, pred_arr)
            kendall = float(kendall_value) if kendall_value is not None else float("nan")
            spearman = float(spearman_value) if spearman_value is not None else float("nan")
        except ImportError:
            note = "scipy_unavailable"
        except Exception:  # pragma: no cover - defensive: bad/constant inputs
            note = "correlation_failed"
    return {"kendall": kendall, "spearman": spearman, "note": note}


def policy_rank_metrics(
    true_score: torch.Tensor | np.ndarray,
    pred_score: torch.Tensor | np.ndarray,
    max_points: int = POLICY_MAX_POINTS,
    seed: int = POLICY_SUBSAMPLE_SEED,
) -> dict[str, Any]:
    """Flattened rank-correlation between all predicted and true output values.

    Every output cell is treated as a point; the model is scored on how well it
    preserves the global ordering of outcomes. The flattened arrays are
    deterministically subsampled to ``max_points`` to keep Kendall's tau tractable
    on high output dimensions.
    """
    true_arr = _to_numpy(true_score).reshape(-1)
    pred_arr = _to_numpy(pred_score).reshape(-1)

    if 0 < max_points < true_arr.size and true_arr.size == pred_arr.size:
        generator = np.random.default_rng(seed)
        selection = generator.choice(true_arr.size, size=max_points, replace=False)
        true_arr = true_arr[selection]
        pred_arr = pred_arr[selection]

    correlations = _rank_correlations(true_arr, pred_arr)
    return {
        "Policy Kendall Tau": correlations["kendall"],
        "Policy Spearman": correlations["spearman"],
        "Policy Metrics Note": correlations["note"],
    }


def policy_aggregate_rank_metrics(
    true_score: torch.Tensor | np.ndarray,
    pred_score: torch.Tensor | np.ndarray,
) -> dict[str, Any]:
    """Rank-correlation of per-sample aggregate policy scores.

    Each sample (row) is reduced to a single scalar policy score by summing across
    its output columns, then the true and predicted scores are rank-correlated
    across samples. This captures how well the surrogate ranks whole
    scenarios/policies, rather than individual output cells.
    """
    true_matrix = _to_numpy(true_score)
    pred_matrix = _to_numpy(pred_score)
    true_matrix = np.atleast_2d(true_matrix)
    pred_matrix = np.atleast_2d(pred_matrix)

    true_totals = true_matrix.sum(axis=1).reshape(-1)
    pred_totals = pred_matrix.sum(axis=1).reshape(-1)

    correlations = _rank_correlations(true_totals, pred_totals)
    return {
        "Policy Aggregate Kendall Tau": correlations["kendall"],
        "Policy Aggregate Spearman": correlations["spearman"],
        "Policy Aggregate Metrics Note": correlations["note"],
    }


def compute_policy_metrics(
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    max_points: int = POLICY_MAX_POINTS,
    seed: int = POLICY_SUBSAMPLE_SEED,
) -> dict[str, Any]:
    """Combined flattened + per-sample-aggregate policy rank metrics.

    Returns a flat dict merging :func:`policy_rank_metrics` (every output cell) and
    :func:`policy_aggregate_rank_metrics` (per-sample summed scores) so the whole
    block can be merged straight into a split's regression-metric dict.
    """
    metrics = policy_rank_metrics(targets, predictions, max_points=max_points, seed=seed)
    metrics.update(policy_aggregate_rank_metrics(targets, predictions))
    return metrics


def add_policy_metrics(
    metrics: dict[str, Any],
    predictions: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    head_slices: dict[str, tuple[int, int]] | None = None,
    max_points: int = POLICY_MAX_POINTS,
    seed: int = POLICY_SUBSAMPLE_SEED,
) -> dict[str, Any]:
    """Merge policy metrics into an existing regression-metrics dict in place.

    For single-head runs the metrics are merged at the top level. For multi-head
    runs (``head_slices`` provided, ``metrics`` shaped ``{"Overall":..., "Per
    Head":...}``) the overall metrics use the full arrays and each head is scored
    on its contiguous output slice, mirroring :func:`evaluate_regression_multihead`.
    """
    if head_slices is None:
        metrics.update(
            compute_policy_metrics(predictions, targets, max_points=max_points, seed=seed)
        )
        return metrics

    metrics["Overall"].update(
        compute_policy_metrics(predictions, targets, max_points=max_points, seed=seed)
    )
    for name, (start, stop) in head_slices.items():
        metrics["Per Head"][name].update(
            compute_policy_metrics(
                predictions[:, start:stop],
                targets[:, start:stop],
                max_points=max_points,
                seed=seed,
            )
        )
    return metrics
