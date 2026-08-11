"""Sparse variational Gaussian-process surrogate (GPyTorch).

Each output column is modelled as an independent batched GP task sharing one set
of inducing inputs. GPyTorch is imported lazily so the Auto and VAE families keep
working in environments where GPyTorch is not installed.

Caveats for CPU-only runs:
- GPyTorch must be available inside the execution container.
- Cost scales with num_tasks (== output_dim) times num_inducing_points^2. High
  output dimensions (e.g. delay with ~800 columns) can be slow on CPU; start
  from a small profile (few inducing points) before scaling up.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import DataLoader

from src.config.schema import GPProfileConfig, TrainLoopConfig
from src.eval.metrics import multihead_metrics_from_arrays
from src.models.base import SurrogateModel
from src.train.loop import EarlyStopping, select_device, set_random_seed
from src.train.transform import TensorStandardizer


_GP_CLASSES: tuple[Any, Any, Any, Any] | None = None


def _gp_classes() -> tuple[Any, Any, Any, Any]:
    """Import GPyTorch on demand and build the batched GP module classes."""

    global _GP_CLASSES
    if _GP_CLASSES is not None:
        return _GP_CLASSES

    try:
        import gpytorch
        import torch.nn.functional as functional
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "The GP surrogate requires the 'gpytorch' package, which is not installed "
            "in this environment."
        ) from error

    class LMCGP(gpytorch.models.ApproximateGP):
        """Latent multi-output GP: a few latent SVGPs mixed into all tasks.

        Only ``num_latents`` independent GPs are evaluated (the batch dimension),
        and the ``num_tasks`` outputs are produced by a learned linear combination
        via LMCVariationalStrategy. Memory scales with ``num_latents`` rather than
        ``num_tasks``, which is essential for high output dimensions.
        """

        def __init__(
            self,
            num_tasks: int,
            num_latents: int,
            input_dim: int,
            inducing_points: torch.Tensor,
            kernel_type: str,
            use_ard: bool,
        ) -> None:
            num_tasks = int(num_tasks)
            num_latents = int(num_latents)
            input_dim = int(input_dim)
            num_inducing = int(inducing_points.size(-2))
            batch_shape = torch.Size([num_latents])
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing,
                batch_shape=batch_shape,
            )
            base_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            )
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                base_strategy,
                num_tasks=num_tasks,
                num_latents=num_latents,
                latent_dim=-1,
            )
            super().__init__(variational_strategy)

            self.num_tasks = num_tasks
            self.num_latents = num_latents
            self.input_dim = input_dim
            self.num_inducing_points = num_inducing
            self.kernel_type = str(kernel_type)
            self.use_ard = bool(use_ard)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

            ard_num_dims = input_dim if self.use_ard else None
            if self.kernel_type == "matern":
                base_kernel = gpytorch.kernels.MaternKernel(
                    nu=1.5, batch_shape=batch_shape, ard_num_dims=ard_num_dims
                )
            elif self.kernel_type == "rbf":
                base_kernel = gpytorch.kernels.RBFKernel(
                    batch_shape=batch_shape, ard_num_dims=ard_num_dims
                )
            elif self.kernel_type == "linear":
                base_kernel = gpytorch.kernels.LinearKernel(
                    batch_shape=batch_shape, ard_num_dims=ard_num_dims
                )
            else:
                raise ValueError(
                    f"Unknown kernel_type={self.kernel_type!r}; expected matern, rbf, or linear"
                )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                base_kernel, batch_shape=batch_shape
            )

        def forward(self, values: torch.Tensor):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(values),
                self.covar_module(values),
            )

    class SoftplusMultitaskGaussianLikelihood(
        gpytorch.likelihoods.MultitaskGaussianLikelihood
    ):
        """Multitask Gaussian likelihood with a non-negative predictive mean."""

        def forward(self, function_samples, *args, **kwargs):
            base = super().forward(function_samples, *args, **kwargs)
            return base.__class__(
                functional.softplus(base.mean),
                base.lazy_covariance_matrix,
            )

    _GP_CLASSES = (
        gpytorch,
        LMCGP,
        SoftplusMultitaskGaussianLikelihood,
        gpytorch.likelihoods.MultitaskGaussianLikelihood,
    )
    return _GP_CLASSES


def deduplicate_rows(
    pool: torch.Tensor,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Drop (near-)duplicate rows from an inducing-point pool.

    Duplicate or near-identical standardized rows (common with sparse/one-hot or
    zero-variance features) make the inducing-inducing covariance rank-deficient,
    which causes the Cholesky factor of ``K_uu`` to fail as not positive definite.
    Rows are quantized to ``tolerance`` and only the first occurrence of each
    unique quantized row is kept, preserving order.
    """
    if pool.ndim != 2:
        raise ValueError(f"pool must be [N,D], got {tuple(pool.shape)}")
    if pool.shape[0] <= 1 or tolerance <= 0:
        return pool
    quantized = torch.round(pool.detach().cpu() / tolerance)
    # Keep the first occurrence of each unique quantized row, preserving order.
    seen: set[tuple[float, ...]] = set()
    keep: list[int] = []
    for row_index in range(quantized.shape[0]):
        key = tuple(quantized[row_index].tolist())
        if key not in seen:
            seen.add(key)
            keep.append(row_index)
    if len(keep) == pool.shape[0]:
        return pool
    keep_tensor = torch.tensor(keep, dtype=torch.long, device=pool.device)
    return pool.index_select(0, keep_tensor)


def make_inducing_points(
    x_train: torch.Tensor,
    num_inducing_points: int,
    seed: int,
    num_latents: int,
) -> torch.Tensor:
    # Returns [num_latents, M, D]. Each latent GP gets a different random subset so
    # the latents are not identical at initialization (learn_inducing_locations
    # separates them further during training).
    if x_train.ndim != 2:
        raise ValueError(f"x_train must be [N,D], got {tuple(x_train.shape)}")
    n = int(x_train.shape[0])
    m = min(int(num_inducing_points), n)
    if m <= 0:
        raise ValueError("num_inducing_points must be positive and x_train must be non-empty")
    num_latents = int(num_latents)
    latents: list[torch.Tensor] = []
    for latent in range(num_latents):
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + latent)
        indices = torch.arange(n) if n == m else torch.randperm(n, generator=generator)[:m]
        latents.append(x_train[indices, :].float().cpu())
    return torch.stack(latents, dim=0).contiguous()


class GPSurrogate(SurrogateModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        profile: GPProfileConfig,
        head_slices: Mapping[str, tuple[int, int]] | None = None,
        head_output_dims: Mapping[str, int] | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_tasks = int(output_dim)
        self.num_latents = int(profile.num_latents)
        self.profile = profile
        # Multi-head metadata. When set, this GP is one shared LMC over all
        # concatenated tasks and ``head_slices`` partition the task axis so
        # metrics/results can be reported per component, exactly like the
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
        self.standardizer: TensorStandardizer | None = None
        self.device = "cpu"
        self._model: Any = None
        self._likelihood: Any = None

    def _collect_inducing_pool(
        self,
        loader: DataLoader,
        standardizer: TensorStandardizer,
        device: str,
        pool_size: int,
    ) -> torch.Tensor:
        # Gather up to pool_size standardized rows for inducing-point selection.
        # Only a bounded number of rows is materialized so datasets with many rows
        # or very high input dimensions do not exhaust memory.
        dataset = getattr(loader, "dataset", None)
        materialized = (
            dataset.materialized_tensors()
            if hasattr(dataset, "materialized_tensors")
            else None
        )
        if materialized is not None:
            pool = materialized[0][:pool_size].to(device)
            return standardizer.transform_inputs(pool).detach()
        collected: list[torch.Tensor] = []
        rows = 0
        for x_batch, _ in loader:
            collected.append(standardizer.transform_inputs(x_batch.to(device)).detach())
            rows += int(x_batch.shape[0])
            if rows >= pool_size:
                break
        if not collected:
            raise ValueError("Training loader produced no batches")
        return torch.cat(collected, dim=0)[:pool_size]

    def _run_epoch(
        self,
        loader: DataLoader,
        standardizer: TensorStandardizer,
        device: str,
        optimizer: torch.optim.Optimizer | None,
        mll: Any,
        gradient_clip: float | None,
    ) -> float:
        training = optimizer is not None
        self._model.train(training)
        self._likelihood.train(training)
        total_loss = 0.0
        total_rows = 0
        gpytorch, _, _, _ = _gp_classes()
        context = torch.enable_grad() if training else torch.no_grad()
        # Loosen the Cholesky jitter tolerance and allow more retries so the
        # inducing-inducing covariance factorizes even when it is poorly
        # conditioned, instead of raising NotPSDError at jitter 1e-6.
        with context, gpytorch.settings.cholesky_jitter(
            float_value=1e-4, double_value=1e-6
        ), gpytorch.settings.cholesky_max_tries(10):
            for x_batch, y_batch in loader:
                x_scaled = standardizer.transform_inputs(x_batch.to(device))
                y_scaled = standardizer.transform_targets(y_batch.to(device))
                targets = y_scaled.contiguous()
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                output = self._model(x_scaled)
                loss = (-mll(output, targets)).mean()
                if optimizer is not None:
                    loss.backward()
                    if gradient_clip is not None:
                        torch.nn.utils.clip_grad_norm_(
                            list(self._model.parameters())
                            + list(self._likelihood.parameters()),
                            gradient_clip,
                        )
                    optimizer.step()
                batch_rows = int(x_batch.shape[0])
                total_loss += float(loss.item()) * batch_rows
                total_rows += batch_rows
        if total_rows == 0:
            raise ValueError("Loader produced no batches")
        return total_loss / total_rows

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
        gpytorch, batched_gp_cls, softplus_cls, gaussian_cls = _gp_classes()

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
        self.standardizer.move(device)

        dataset_length = len(training_dataset) if training_dataset is not None else 0
        if dataset_length <= 0:
            raise ValueError("GP training requires a sized training dataset")
        num_data = int(dataset_length)

        # Only a small pool of rows is needed to initialize inducing locations.
        budget_elements = 50_000_000
        rows_by_budget = max(
            self.profile.num_inducing_points,
            budget_elements // max(1, self.input_dim),
        )
        pool_size = min(
            num_data,
            max(self.profile.num_inducing_points, min(4096, rows_by_budget)),
        )
        pool = self._collect_inducing_pool(
            train_loader, self.standardizer, device, pool_size
        )
        # Remove near-duplicate rows so the inducing points are distinct; identical
        # rows make K_uu rank-deficient and its Cholesky factor fails (NotPSDError).
        pool = deduplicate_rows(pool)
        inducing = make_inducing_points(
            pool,
            num_inducing_points=self.profile.num_inducing_points,
            seed=self.profile.inducing_seed,
            num_latents=self.num_latents,
        ).to(device)
        del pool

        self._model = batched_gp_cls(
            self.num_tasks,
            self.num_latents,
            self.input_dim,
            inducing,
            self.profile.kernel_type,
            self.profile.use_ard,
        ).to(device)
        likelihood_cls = softplus_cls if self.profile.use_softplus_likelihood else gaussian_cls
        self._likelihood = likelihood_cls(
            num_tasks=self.num_tasks,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
        ).to(device)

        mll = gpytorch.mlls.VariationalELBO(
            self._likelihood, self._model, num_data=num_data
        )
        optimizer = torch.optim.Adam(
            list(self._model.parameters()) + list(self._likelihood.parameters()),
            lr=self.profile.learning_rate,
        )

        stopper = EarlyStopping(config.patience, config.minimum_delta)
        history: list[dict[str, float | int]] = []
        best_validation_loss = float("inf")
        best_epoch = 0
        best_state: dict[str, Any] | None = None

        for epoch in range(1, config.epochs + 1):
            train_loss = self._run_epoch(
                train_loader, self.standardizer, device, optimizer, mll, config.gradient_clip
            )
            validation_loss = self._run_epoch(
                validation_loader, self.standardizer, device, None, mll, None
            )
            history.append(
                {
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Validation Loss": validation_loss,
                    "Learning Rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if epoch == 1 or epoch % config.print_every == 0 or epoch == config.epochs:
                print(
                    f"[training] epoch={epoch}/{config.epochs} "
                    f"train_loss={train_loss:.8f} "
                    f"validation_loss={validation_loss:.8f}",
                    flush=True,
                )
            if validation_loss < best_validation_loss:
                best_validation_loss = float(validation_loss)
                best_epoch = epoch
                best_state = {
                    "model": copy.deepcopy(self._model.state_dict()),
                    "likelihood": copy.deepcopy(self._likelihood.state_dict()),
                }
            if periodic_save_callback is not None and epoch % 25 == 0:
                periodic_save_callback(epoch)
            if stopper.step(validation_loss):
                print(f"[training] early stopping at epoch {epoch}", flush=True)
                break

        if best_state is None:
            raise RuntimeError("Training finished without a valid checkpoint")

        self._model.load_state_dict(best_state["model"])
        self._likelihood.load_state_dict(best_state["likelihood"])
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

    def predict(
        self,
        loader: DataLoader,
        max_batches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.standardizer is None or self._model is None:
            raise RuntimeError("The model must be fitted or loaded before prediction")
        gpytorch, _, _, _ = _gp_classes()
        self._model.eval()
        self._likelihood.eval()
        self.standardizer.move(self.device)
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(
            float_value=1e-4, double_value=1e-6
        ), gpytorch.settings.cholesky_max_tries(10):
            for batch_index, (x_batch, y_batch) in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                x_scaled = self.standardizer.transform_inputs(x_batch.to(self.device))
                output = self._likelihood(self._model(x_scaled))
                mean_scaled = output.mean.contiguous()
                prediction = self.standardizer.inverse_targets(mean_scaled)
                predictions.append(prediction.detach().cpu())
                targets.append(y_batch.detach().cpu())
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
        if self.standardizer is None or self._model is None:
            raise RuntimeError("The model must be fitted or loaded before saving")
        inducing = (
            self._model.variational_strategy.base_variational_strategy.inducing_points.detach()
            .cpu()
        )
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        is_multi = self.head_slices is not None
        torch.save(
            {
                "family": "GP",
                "mode": "multi" if is_multi else "single",
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "num_tasks": self.num_tasks,
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
                "inducing_points": inducing,
                "model_state": self._model.state_dict(),
                "likelihood_state": self._likelihood.state_dict(),
                "standardizer_state": self.standardizer.state_dict(),
                "metadata": metadata or {},
            },
            output_path,
            pickle_protocol=4,
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        gpytorch, batched_gp_cls, softplus_cls, gaussian_cls = _gp_classes()
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
        inducing = checkpoint["inducing_points"].to(self.device)
        self._model = batched_gp_cls(
            self.num_tasks,
            self.num_latents,
            self.input_dim,
            inducing,
            self.profile.kernel_type,
            self.profile.use_ard,
        ).to(self.device)
        likelihood_cls = softplus_cls if self.profile.use_softplus_likelihood else gaussian_cls
        self._likelihood = likelihood_cls(
            num_tasks=self.num_tasks,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4),
        ).to(self.device)
        self._model.load_state_dict(checkpoint["model_state"])
        self._likelihood.load_state_dict(checkpoint["likelihood_state"])
        standardizer_state = checkpoint["standardizer_state"]
        self.standardizer = TensorStandardizer(
            standardize_inputs=bool(standardizer_state["standardize_inputs"]),
            standardize_targets=bool(standardizer_state["standardize_targets"]),
            epsilon=float(standardizer_state["epsilon"]),
        )
        self.standardizer.load_state_dict(standardizer_state)
        head_slices = checkpoint.get("head_slices")
        if head_slices is not None:
            self.head_slices = {
                str(name): (int(bounds[0]), int(bounds[1]))
                for name, bounds in head_slices.items()
            }
        head_output_dims = checkpoint.get("head_output_dims")
        if head_output_dims is not None:
            self.head_output_dims = {str(name): int(dim) for name, dim in head_output_dims.items()}
        return dict(checkpoint.get("metadata", {}))
