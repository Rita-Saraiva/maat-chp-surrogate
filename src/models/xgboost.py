"""Gradient-boosted decision-tree surrogate (XGBoost).

All output columns are predicted by a single native multi-output model
(``multi_strategy="multi_output_tree"``), so the training cost does not scale
linearly with the number of target columns. XGBoost is imported lazily so the
other families keep working in environments where XGBoost is not installed.

Design notes:
- Input/target standardization follows ``config.standardization`` (like the neural
  families). Statistics are fit once from the training data; inputs and targets are
  standardized as they stream into the matrices, and predictions are inverse-
  transformed so metrics stay in the original target space. Trees are
  scale-invariant, so this mainly aligns XGB with the other families rather than
  materially changing accuracy.
- Training and validation data are streamed into quantized ``QuantileDMatrix``
  objects via a ``DataIter`` (one batch at a time), so the full dense dataset is
  never held in RAM. This keeps memory bounded on high-dimensional chunked
  datasets that do not fit in memory as dense float32.
- When even the quantized matrix would not fit in RAM, downsample by exporting
  ``XGB_KEEP_FRACTION`` (and optionally ``XGB_VAL_KEEP_FRACTION``) in (0, 1];
  these are estimated up front by ``tools/estimate_xgb_subsample.py`` (run once in
  a separate PBS step). Without them the model trains on the full data.
  Downsampling is uniform per batch so every shard/scenario stays proportionally
  represented (unlike front-truncating MAX_SAMPLES), and evaluation always runs on
  the full splits.
- Native multi-output over many columns can still be time-heavy on CPU; start
  from a small profile, and prefer ``one_output_per_tree`` when the output
  dimension is large (vector-leaf histograms scale with the output dimension).
- Requires XGBoost >= 2.0 in the execution container for ``multi_output_tree``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config.schema import TrainLoopConfig, XGBProfileConfig
from src.eval.metrics import multihead_metrics_from_arrays
from src.models.base import SurrogateModel
from src.train.loop import select_device, set_random_seed
from src.train.transform import TensorStandardizer


def _xgb() -> Any:
    """Import XGBoost on demand with a clear error when it is unavailable."""

    try:
        import xgboost
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "The XGBoost surrogate requires the 'xgboost' package, which is not "
            "installed in this environment."
        ) from error
    return xgboost


_DATA_ITER_CLS: Any = None


def _data_iter_class() -> Any:
    """Build (once) an XGBoost ``DataIter`` that streams a torch DataLoader.

    XGBoost pulls one batch at a time via this iterator and stores only the
    compact quantized (ellpack) representation, so the full dense dataset is never
    held in RAM. This is what lets high-dimensional datasets that do not fit in
    memory as dense float32 still train.
    """

    global _DATA_ITER_CLS
    if _DATA_ITER_CLS is not None:
        return _DATA_ITER_CLS
    xgboost = _xgb()

    class _LoaderDataIter(xgboost.DataIter):
        """Streams identical ``(X, y)`` batches on every pass.

        ``QuantileDMatrix`` iterates its source more than once (a sketch pass then
        a fill pass) and calls ``reset()`` between passes. The training loaders
        reshuffle on every iteration, which would hand XGBoost different batches
        per pass and corrupt its internal per-batch bookkeeping. We therefore
        iterate a fresh non-shuffling loader over the same dataset; row order does
        not affect tree construction.

        When ``keep_fraction < 1`` each batch is uniformly downsampled to a
        deterministic subset (seeded by batch index) so the quantized matrix stays
        within the memory budget. Because every batch keeps the same fraction, the
        sample is proportional across all shards/scenarios rather than truncating
        from the front, and it is identical on every pass so the multi-pass build
        stays consistent.
        """

        def __init__(
            self,
            loader: DataLoader,
            keep_fraction: float = 1.0,
            seed: int = 0,
            standardizer: TensorStandardizer | None = None,
        ) -> None:
            from src.data.training import create_training_loader

            batch_sampler = getattr(loader, "batch_sampler", None)
            batch_size = int(
                getattr(batch_sampler, "batch_size", None)
                or getattr(loader, "batch_size", None)
                or 1024
            )
            # Stream single-threaded: worker prefetch would hold several large
            # shards (e.g. econ_combined ~1.76 GiB each) in memory at once and OOM
            # the node. Row order is irrelevant to tree building, so one shard at a
            # time is both memory-safe and correct.
            self._loader = create_training_loader(
                loader.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            self._keep_fraction = float(min(1.0, max(0.0, keep_fraction)))
            self._seed = int(seed)
            self._standardizer = standardizer
            self._iterator: Any = None
            self._batch_index = 0
            super().__init__()

        def reset(self) -> None:
            self._iterator = iter(self._loader)
            self._batch_index = 0

        def _subsample(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            n = x.shape[0]
            keep = max(1, int(round(self._keep_fraction * n)))
            if keep >= n:
                return x, y
            rng = np.random.default_rng(self._seed + self._batch_index)
            selected = np.sort(rng.choice(n, size=keep, replace=False))
            return x[selected], y[selected]

        def next(self, input_data: Any) -> int:
            if self._iterator is None:
                self._iterator = iter(self._loader)
            try:
                x_batch, y_batch = next(self._iterator)
            except StopIteration:
                return 0
            x_np = np.ascontiguousarray(x_batch.detach().cpu().numpy(), dtype=np.float32)
            y_np = np.ascontiguousarray(y_batch.detach().cpu().numpy(), dtype=np.float32)
            if self._keep_fraction < 1.0:
                x_np, y_np = self._subsample(x_np, y_np)
                x_np = np.ascontiguousarray(x_np, dtype=np.float32)
                y_np = np.ascontiguousarray(y_np, dtype=np.float32)
            if self._standardizer is not None:
                x_np = np.ascontiguousarray(
                    self._standardizer.transform_inputs(torch.from_numpy(x_np)).numpy(),
                    dtype=np.float32,
                )
                y_np = np.ascontiguousarray(
                    self._standardizer.transform_targets(torch.from_numpy(y_np)).numpy(),
                    dtype=np.float32,
                )
            self._batch_index += 1
            input_data(data=x_np, label=y_np)
            return 1

    _DATA_ITER_CLS = _LoaderDataIter
    return _DATA_ITER_CLS


def _keep_fraction_from_env(name: str) -> float:
    """Read a keep-fraction from environment variable ``name``.

    Values are clamped to ``(0, 1]``. A missing, malformed, or non-positive value
    yields ``1.0`` so a training job submitted without an estimate simply trains on
    the full data. The fractions themselves are produced up front by
    ``tools/estimate_xgb_subsample.py``.
    """

    raw = os.getenv(name)
    if not raw:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    if not value > 0.0:
        return 1.0
    return min(1.0, value)


class XGBSurrogate(SurrogateModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        profile: XGBProfileConfig,
        head_slices: Mapping[str, tuple[int, int]] | None = None,
        head_output_dims: Mapping[str, int] | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.profile = profile
        # Multi-head metadata. When set, this is a single native multi-output model
        # over all concatenated columns and ``head_slices`` partition the output
        # axis so metrics/results can be reported per component, exactly like the
        # shared-trunk families. ``None`` keeps the single-head behaviour.
        self.head_slices: dict[str, tuple[int, int]] | None = (
            {str(name): (int(bounds[0]), int(bounds[1])) for name, bounds in head_slices.items()}
            if head_slices is not None
            else None
        )
        if head_output_dims is not None:
            self.head_output_dims: dict[str, int] | None = {
                str(name): int(dim) for name, dim in head_output_dims.items()
            }
        elif self.head_slices is not None:
            self.head_output_dims = {
                name: stop - start for name, (start, stop) in self.head_slices.items()
            }
        else:
            self.head_output_dims = None
        self.device = "cpu"
        self._booster: Any = None
        self._best_iteration = 0
        self.standardizer: TensorStandardizer | None = None

    def _params(self, random_state: int, device: str) -> dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "tree_method": str(self.profile.tree_method),
            "multi_strategy": str(self.profile.multi_strategy),
            "max_depth": int(self.profile.max_depth),
            "learning_rate": float(self.profile.learning_rate),
            "subsample": float(self.profile.subsample),
            "colsample_bytree": float(self.profile.colsample_bytree),
            "min_child_weight": float(self.profile.min_child_weight),
            "reg_lambda": float(self.profile.reg_lambda),
            "reg_alpha": float(self.profile.reg_alpha),
            "gamma": float(self.profile.gamma),
            "max_bin": int(self.profile.max_bin),
            "nthread": int(self.profile.n_jobs),
            "seed": int(random_state),
            "device": device,
        }

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        config: TrainLoopConfig,
        checkpoint_path: str | Path,
        periodic_save_callback: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        set_random_seed(config.seed, config.deterministic)
        device = select_device(config.device)
        self.device = device
        xgboost = _xgb()
        iterator_cls = _data_iter_class()

        # Standardization follows config.standardization (like the NN families).
        # Statistics are fit once from a non-shuffling, single-worker pass so worker
        # prefetch never holds several large shards at once; inputs/targets are then
        # standardized as they stream into the matrices and inverse-transformed at
        # prediction time so metrics stay in the original target space.
        from src.data.training import create_training_loader

        self.standardizer = TensorStandardizer(
            standardize_inputs=config.standardization.inputs,
            standardize_targets=config.standardization.targets,
            epsilon=config.standardization.epsilon,
        )
        stats_batch_size = int(
            getattr(getattr(train_loader, "batch_sampler", None), "batch_size", None)
            or getattr(train_loader, "batch_size", None)
            or 1024
        )
        stats_loader = create_training_loader(
            train_loader.dataset,
            batch_size=stats_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        print("[training] fitting standardization statistics", flush=True)
        self.standardizer.fit(stats_loader)

        # Stream the loaders into quantized matrices; the dense data is never fully
        # materialized in RAM. Validation reuses the training quantile boundaries.
        # Keep-fractions are estimated up front by tools/estimate_xgb_subsample.py
        # and passed in via the environment; absent an estimate the model trains on
        # the full data. Downsampling is uniform per batch inside the DataIter, so
        # every shard/scenario stays proportionally represented.
        train_rows = len(train_loader.dataset)
        val_rows = len(validation_loader.dataset)
        f_train = _keep_fraction_from_env("XGB_KEEP_FRACTION")
        f_val = _keep_fraction_from_env("XGB_VAL_KEEP_FRACTION")
        kept_train = max(1, int(round(f_train * train_rows)))
        kept_val = max(1, int(round(f_val * val_rows)))
        src_train = "XGB_KEEP_FRACTION" if f_train < 1.0 else "full"
        src_val = "XGB_VAL_KEEP_FRACTION" if f_val < 1.0 else "full"
        print(
            f"[training] keep_train={f_train:.4f} ({kept_train}/{train_rows}, {src_train}) "
            f"keep_val={f_val:.4f} ({kept_val}/{val_rows}, {src_val})",
            flush=True,
        )
        if f_train < 1.0 or f_val < 1.0:
            print(
                "[training] uniformly subsampling across all shards to fit the "
                "quantized matrix in RAM (evaluation still uses the full splits)",
                flush=True,
            )
        print("[training] building quantized train/validation matrices", flush=True)
        max_bin = int(self.profile.max_bin)
        dtrain = xgboost.QuantileDMatrix(
            iterator_cls(train_loader, f_train, config.seed, standardizer=self.standardizer),
            max_bin=max_bin,
        )
        dvalidation = xgboost.QuantileDMatrix(
            iterator_cls(
                validation_loader, f_val, config.seed + 1, standardizer=self.standardizer
            ),
            max_bin=max_bin,
            ref=dtrain,
        )

        early_stopping_rounds = self.profile.early_stopping_rounds
        params = self._params(random_state=config.seed, device=device)
        print(
            f"[training] fitting XGBoost n_estimators={self.profile.n_estimators} "
            f"max_depth={self.profile.max_depth} multi_strategy={self.profile.multi_strategy}",
            flush=True,
        )
        evals_result: dict[str, Any] = {}
        self._booster = xgboost.train(
            params,
            dtrain,
            num_boost_round=int(self.profile.n_estimators),
            evals=[(dtrain, "train"), (dvalidation, "validation")],
            evals_result=evals_result,
            early_stopping_rounds=(
                int(early_stopping_rounds) if early_stopping_rounds is not None else None
            ),
            verbose_eval=False,
        )

        best_iteration = getattr(self._booster, "best_iteration", None)
        if best_iteration is None:
            best_iteration = int(self.profile.n_estimators) - 1
        self._best_iteration = int(best_iteration)

        train_metric = evals_result.get("train", {})
        val_metric = evals_result.get("validation", {})
        metric_name = next(iter(val_metric), None)
        train_curve = train_metric.get(metric_name, []) if metric_name else []
        val_curve = val_metric.get(metric_name, []) if metric_name else []

        history: list[dict[str, float | int]] = []
        for index, validation_loss in enumerate(val_curve):
            train_loss = float(train_curve[index]) if index < len(train_curve) else float("nan")
            history.append(
                {
                    "Epoch": index + 1,
                    "Train Loss": float(train_loss),
                    "Validation Loss": float(validation_loss),
                    "Learning Rate": float(self.profile.learning_rate),
                }
            )
            if index == 0 or (index + 1) % config.print_every == 0 or index == len(val_curve) - 1:
                print(
                    f"[training] round={index + 1}/{len(val_curve)} "
                    f"train_{metric_name}={train_loss:.8f} "
                    f"validation_{metric_name}={float(validation_loss):.8f}",
                    flush=True,
                )

        best_validation_loss = (
            float(val_curve[self._best_iteration])
            if 0 <= self._best_iteration < len(val_curve)
            else (float(min(val_curve)) if val_curve else float("nan"))
        )
        print(
            f"[training] best iteration={self._best_iteration + 1} "
            f"validation_{metric_name}={best_validation_loss:.8f}",
            flush=True,
        )

        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "booster_raw": bytes(self._booster.save_raw(raw_format="ubj")),
                "best_iteration": self._best_iteration,
                "standardizer_state": self.standardizer.state_dict(),
            },
            checkpoint,
            pickle_protocol=4,
        )

        return {
            "Device": device,
            "Best Epoch": self._best_iteration + 1,
            "Best Validation Loss": best_validation_loss,
            "Epochs Trained": len(history),
            "History": history,
        }

    def _predict_array(self, x_values: np.ndarray) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("The model must be fitted or loaded before prediction")
        x_arr = np.ascontiguousarray(x_values, dtype=np.float32)
        if self.standardizer is not None:
            x_arr = np.ascontiguousarray(
                self.standardizer.transform_inputs(torch.from_numpy(x_arr)).numpy(),
                dtype=np.float32,
            )
        predictions = self._booster.inplace_predict(
            x_arr,
            iteration_range=(0, self._best_iteration + 1),
        )
        predictions = np.asarray(predictions, dtype=np.float32)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if self.standardizer is not None:
            predictions = np.ascontiguousarray(
                self.standardizer.inverse_targets(torch.from_numpy(predictions)).numpy(),
                dtype=np.float32,
            )
        return predictions

    def predict(
        self,
        loader: DataLoader,
        max_batches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._booster is None:
            raise RuntimeError("The model must be fitted or loaded before prediction")
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for batch_index, (x_batch, y_batch) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch_predictions = self._predict_array(x_batch.detach().cpu().numpy())
            predictions.append(torch.from_numpy(batch_predictions))
            targets.append(y_batch.detach().cpu().float())
        if not predictions:
            raise ValueError("Prediction loader produced no batches")
        return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)

    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        predictions, targets = self.predict(loader)
        if self.head_slices is not None:
            return multihead_metrics_from_arrays(targets, predictions, self.head_slices)
        errors = predictions - targets
        count = int(targets.numel())
        squared_error_sum = float(errors.square().sum().item())
        absolute_error_sum = float(errors.abs().sum().item())
        maximum_absolute_error = float(errors.abs().max().item())
        mean_target = float(targets.mean().item())
        total_square = float((targets - mean_target).square().sum().item())
        r2 = 1.0 - squared_error_sum / total_square if total_square > 0.0 else float("nan")
        return {
            "MSE": squared_error_sum / count,
            "RMSE": math.sqrt(squared_error_sum / count),
            "MAE": absolute_error_sum / count,
            "R2": r2,
            "Max Absolute Error": maximum_absolute_error,
            "Values": count,
        }

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        if self._booster is None:
            raise RuntimeError("The model must be fitted or loaded before saving")
        booster_raw = bytes(self._booster.save_raw(raw_format="ubj"))
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        is_multi = self.head_slices is not None
        torch.save(
            {
                "family": "XGB",
                "mode": "multi" if is_multi else "single",
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "best_iteration": self._best_iteration,
                "profile": self.profile.model_dump(mode="python"),
                "target_names": (
                    list(self.head_output_dims.keys())
                    if self.head_output_dims is not None
                    else None
                ),
                "head_output_dims": (
                    dict(self.head_output_dims) if self.head_output_dims is not None else None
                ),
                "head_slices": (
                    {name: list(bounds) for name, bounds in self.head_slices.items()}
                    if is_multi
                    else None
                ),
                "booster_raw": booster_raw,
                "standardizer_state": (
                    self.standardizer.state_dict() if self.standardizer is not None else None
                ),
                "metadata": metadata or {},
            },
            output_path,
            pickle_protocol=4,
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        xgboost = _xgb()
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        if int(checkpoint["input_dim"]) != self.input_dim:
            raise ValueError(
                f"Checkpoint input_dim={checkpoint['input_dim']} does not match model input_dim={self.input_dim}"
            )
        if int(checkpoint["output_dim"]) != self.output_dim:
            raise ValueError(
                f"Checkpoint output_dim={checkpoint['output_dim']} does not match model output_dim={self.output_dim}"
            )
        self.device = select_device("auto")
        self._booster = xgboost.Booster()
        self._booster.load_model(bytearray(checkpoint["booster_raw"]))
        self._best_iteration = int(checkpoint.get("best_iteration", 0))
        head_slices = checkpoint.get("head_slices")
        if head_slices is not None:
            self.head_slices = {
                str(name): (int(bounds[0]), int(bounds[1]))
                for name, bounds in head_slices.items()
            }
        head_output_dims = checkpoint.get("head_output_dims")
        if head_output_dims is not None:
            self.head_output_dims = {str(name): int(dim) for name, dim in head_output_dims.items()}
        standardizer_state = checkpoint.get("standardizer_state")
        if standardizer_state is not None:
            self.standardizer = TensorStandardizer(
                standardize_inputs=bool(standardizer_state["standardize_inputs"]),
                standardize_targets=bool(standardizer_state["standardize_targets"]),
                epsilon=float(standardizer_state["epsilon"]),
            )
            self.standardizer.load_state_dict(standardizer_state)
        else:
            # Older checkpoints predate standardization; predict in original units.
            self.standardizer = None
        return dict(checkpoint.get("metadata", {}))
