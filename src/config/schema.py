from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ObjectiveName = Literal["Econ", "QoL"]
DatasetMode = Literal["chunked", "whole"]
SplitName = Literal["train", "val", "test"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SplitConfig(StrictModel):
    train_weight: int = Field(default=7, ge=0)
    val_weight: int = Field(default=1, ge=0)
    test_weight: int = Field(default=2, ge=0)
    shuffle: bool = True
    seed: int = 42

    @model_validator(mode="after")
    def validate_positive_total(self) -> "SplitConfig":
        if self.train_weight + self.val_weight + self.test_weight <= 0:
            raise ValueError("At least one split weight must be positive.")
        return self


class BuildOptions(StrictModel):
    num_regions: int = Field(default=278, gt=0)
    episode_len: int = Field(default=0, ge=0)
    episode_stride: int = Field(default=1, gt=0)
    max_episodes_per_env: int = Field(default=0, ge=0)
    samples_per_shard: int = Field(default=5000, gt=0)
    action_classes: int = Field(default=0, ge=0)
    include_action_onehot: bool = True
    overwrite: bool = False


class BuildSettings(StrictModel):
    project_root: Path
    output_root: Path
    raw_roots: dict[str, Path]
    split: SplitConfig = Field(default_factory=SplitConfig)
    build: BuildOptions = Field(default_factory=BuildOptions)

    def raw_root_for(self, objective: str) -> Path:
        try:
            return self.raw_roots[objective]
        except KeyError as exc:
            raise ValueError(
                f"No raw root configured for objective={objective!r}. "
                f"Configured objectives: {sorted(self.raw_roots)}"
            ) from exc


class SourceConfig(StrictModel):
    source_index: int = Field(ge=0)
    raw_dir: Path
    buffer_name: str = Field(min_length=1)
    rl_config: Path


class BufferAssignment(StrictModel):
    dataset_env_id: int = Field(ge=0)
    split: SplitName
    source_index: int = Field(ge=0)
    source_env_id: int = Field(ge=0)
    source_file: Path
    raw_dir: Path
    buffer_name: str
    rl_config: Path


class DatasetBuildRequest(StrictModel):
    objective: ObjectiveName
    folder_name: str = Field(min_length=1)
    output_dir: Path
    job_id: str = "local"
    sources: list[SourceConfig]
    split: SplitConfig = Field(default_factory=SplitConfig)
    build: BuildOptions = Field(default_factory=BuildOptions)

    @field_validator("folder_name")
    @classmethod
    def clean_folder_name(cls, value: str) -> str:
        cleaned = value.strip().strip(",").replace(" ", "")
        if not cleaned:
            raise ValueError("folder_name cannot be empty.")
        if "/" in cleaned or "\\" in cleaned:
            raise ValueError("folder_name must be a single directory name.")
        return cleaned

    @model_validator(mode="after")
    def require_sources(self) -> "DatasetBuildRequest":
        if not self.sources:
            raise ValueError("At least one source is required.")
        return self


class DataConfig(StrictModel):
    dataset_dir: Path
    split: SplitName = "train"
    mode: DatasetMode = "chunked"
    component: str | None = None
    batch_size: int = Field(default=64, gt=0)
    shuffle: bool = True
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    shard_cache_size: int = Field(default=2, gt=0)


class ModelConfig(StrictModel):
    family: Literal["MLP", "Auto", "VAE", "GP", "XGB"]
    variant: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class TrainConfig(StrictModel):
    epochs: int = Field(default=500, gt=0)
    batch_size: int = Field(default=64, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    patience: int = Field(default=50, ge=0)
    print_every: int = Field(default=25, gt=0)
    seed: int = 42


class RunConfig(StrictModel):
    job_id: str = "local"
    dataset: str
    mode: str
    objective: ObjectiveName
    component: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig = Field(default_factory=TrainConfig)


class SourceRule(StrictModel):
    raw_dir: Path
    default: bool = False
    buffer_contains: str | None = None


class SourceRoots(StrictModel):
    version: int = 1
    objectives: dict[str, list[SourceRule]] = Field(default_factory=dict)

    def resolve(self, objective: str, buffer_name: str) -> Path | None:
        rules = self.objectives.get(objective, [])
        default_rule: SourceRule | None = None
        for rule in rules:
            if rule.buffer_contains and rule.buffer_contains in buffer_name:
                return rule.raw_dir
            if rule.default:
                default_rule = rule
        return default_rule.raw_dir if default_rule else None


# Phase 2 training configuration schemas
from typing import Literal

from pydantic import BaseModel as TrainingBaseModel
from pydantic import ConfigDict as TrainingConfigDict
from pydantic import Field as TrainingField
from pydantic import field_validator as training_field_validator
from pydantic import model_validator as training_model_validator


class DataTrainingConfig(TrainingBaseModel):
    model_config = TrainingConfigDict(extra="forbid")

    dataset: str
    dataset_dir: Path
    loader_mode: Literal["chunked", "whole"] = "chunked"
    component: str
    components: list[str] | None = None
    batch_size: int = TrainingField(default=1024, gt=0)
    num_workers: int = TrainingField(default=0, ge=0)
    pin_memory: bool = True
    max_samples: int | None = TrainingField(default=None, gt=0)
    train_fraction: float | None = TrainingField(default=None, gt=0, le=1)

    @training_field_validator("components")
    @classmethod
    def validate_components(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = [str(name).strip() for name in values if str(name).strip()]
        if not cleaned:
            raise ValueError("components, when provided, must contain at least one name")
        seen: set[str] = set()
        unique: list[str] = []
        for name in cleaned:
            if name not in seen:
                seen.add(name)
                unique.append(name)
        return unique


class MLPProfileConfig(TrainingBaseModel):
    model_config = TrainingConfigDict(extra="forbid")

    hidden_dims: list[int]
    dropout: float = TrainingField(default=0.05, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    layer_norm: bool = False

    @training_field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, values: list[int]) -> list[int]:
        if not values or any(value <= 0 for value in values):
            raise ValueError("hidden_dims must contain positive integers")
        return values


class AutoProfileConfig(TrainingBaseModel):
    """Supervised bottleneck regressor: encoder -> latent -> predictor."""

    model_config = TrainingConfigDict(extra="forbid")

    hidden_dims: list[int]
    latent_dim: int = TrainingField(gt=0)
    dropout: float = TrainingField(default=0.05, ge=0.0, lt=1.0)

    @training_field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, values: list[int]) -> list[int]:
        if not values or any(value <= 0 for value in values):
            raise ValueError("hidden_dims must contain positive integers")
        return values


class VAEProfileConfig(TrainingBaseModel):
    """Variational regressor q(z|x) -> decode(z) -> y with recon + beta*KL loss."""

    model_config = TrainingConfigDict(extra="forbid")

    encoder_hidden_dims: list[int]
    decoder_hidden_dims: list[int] | None = None
    latent_dim: int = TrainingField(default=32, gt=0)
    dropout: float = TrainingField(default=0.05, ge=0.0, lt=1.0)
    beta_start: float = TrainingField(default=0.0, ge=0.0)
    beta_final: float = TrainingField(default=1e-3, ge=0.0)
    beta_warmup_epochs: int = TrainingField(default=30, ge=1)

    @training_field_validator("encoder_hidden_dims")
    @classmethod
    def validate_encoder_hidden_dims(cls, values: list[int]) -> list[int]:
        if not values or any(value <= 0 for value in values):
            raise ValueError("encoder_hidden_dims must contain positive integers")
        return values

    @training_field_validator("decoder_hidden_dims")
    @classmethod
    def validate_decoder_hidden_dims(cls, values: list[int] | None) -> list[int] | None:
        if values is not None and any(value <= 0 for value in values):
            raise ValueError("decoder_hidden_dims must contain positive integers")
        return values


class GPProfileConfig(TrainingBaseModel):
    """Sparse variational GP (GPyTorch) with one batched task per output column."""

    model_config = TrainingConfigDict(extra="forbid")

    num_inducing_points: int = TrainingField(gt=0)
    num_latents: int = TrainingField(default=16, gt=0)
    kernel_type: Literal["matern", "rbf", "linear"] = "matern"
    use_ard: bool = False
    shared_inducing_points: bool = True
    use_softplus_likelihood: bool = True
    learning_rate: float = TrainingField(default=1e-2, gt=0.0)
    eval_batch_size: int = TrainingField(default=256, gt=0)
    inducing_seed: int = 42


class XGBProfileConfig(TrainingBaseModel):
    """Gradient-boosted decision trees (XGBoost) with native multi-output.

    High-dimensional targets are handled by a single native multi-output model
    (``multi_strategy="multi_output_tree"``) rather than one model per column, so
    the cost does not scale linearly with the number of output columns.
    Input/target standardization follows the training ``standardization`` settings
    (fit from the training data, applied as data streams in, predictions
    inverse-transformed); trees are scale-invariant, so this mainly aligns XGB with
    the other families.
    """

    model_config = TrainingConfigDict(extra="forbid")

    n_estimators: int = TrainingField(default=300, gt=0)
    max_depth: int = TrainingField(default=6, gt=0)
    learning_rate: float = TrainingField(default=0.1, gt=0.0)
    subsample: float = TrainingField(default=1.0, gt=0.0, le=1.0)
    colsample_bytree: float = TrainingField(default=1.0, gt=0.0, le=1.0)
    min_child_weight: float = TrainingField(default=1.0, ge=0.0)
    reg_lambda: float = TrainingField(default=1.0, ge=0.0)
    reg_alpha: float = TrainingField(default=0.0, ge=0.0)
    gamma: float = TrainingField(default=0.0, ge=0.0)
    max_bin: int = TrainingField(default=256, gt=0)
    tree_method: Literal["hist", "approx", "exact"] = "hist"
    multi_strategy: Literal["multi_output_tree", "one_output_per_tree"] = (
        "multi_output_tree"
    )
    early_stopping_rounds: int | None = TrainingField(default=20, gt=0)
    n_jobs: int = TrainingField(default=4, gt=0)


class GNNProfileConfig(TrainingBaseModel):
    model_config = TrainingConfigDict(extra="forbid")

    hidden_dim: int = TrainingField(gt=0)
    num_layers: int = TrainingField(default=2, gt=0)
    dropout: float = TrainingField(default=0.05, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    conv_type: Literal["gcn", "sage"] = "gcn"
    residual: bool = True
    layer_norm: bool = True
    add_self_loops: bool = True
    undirected: bool = True
    graph_file: str = "configs/graph/region_graph.yaml"


class ModelTrainingConfig(TrainingBaseModel):
    model_config = TrainingConfigDict(extra="forbid")

    family: Literal["MLP", "Auto", "VAE", "GP", "XGB", "GNN"] = "MLP"
    profile: str
    parameters: (
        MLPProfileConfig
        | AutoProfileConfig
        | VAEProfileConfig
        | GPProfileConfig
        | XGBProfileConfig
        | GNNProfileConfig
    )


class OptimizerConfig(TrainingBaseModel):
    model_config = TrainingConfigDict(extra="forbid")

    name: Literal["adam", "adamw"] = "adamw"
    learning_rate: float = TrainingField(default=1e-3, gt=0.0)
    weight_decay: float = TrainingField(default=1e-5, ge=0.0)


class SchedulerConfig(TrainingBaseModel):
    model_config = TrainingConfigDict(extra="forbid")

    enabled: bool = True
    factor: float = TrainingField(default=0.5, gt=0.0, lt=1.0)
    patience: int = TrainingField(default=5, ge=1)
    minimum_learning_rate: float = TrainingField(default=1e-7, gt=0.0)


class StandardizationConfig(TrainingBaseModel):
    model_config = TrainingConfigDict(extra="forbid")

    inputs: bool = True
    targets: bool = True
    epsilon: float = TrainingField(default=1e-6, gt=0.0)


class TrainLoopConfig(TrainingBaseModel):
    model_config = TrainingConfigDict(extra="forbid")

    epochs: int = TrainingField(default=200, gt=0)
    loss: Literal["mse", "mae", "huber"] = "mse"
    patience: int = TrainingField(default=20, ge=1)
    minimum_delta: float = TrainingField(default=1e-7, ge=0.0)
    print_every: int = TrainingField(default=25, ge=1)
    gradient_clip: float | None = TrainingField(default=None, gt=0.0)
    seed: int = 42
    device: Literal["auto", "cpu", "cuda"] = "auto"
    deterministic: bool = False
    optimizer: OptimizerConfig = TrainingField(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = TrainingField(default_factory=SchedulerConfig)
    standardization: StandardizationConfig = TrainingField(default_factory=StandardizationConfig)


class TrainingRunConfig(TrainingBaseModel):
    model_config = TrainingConfigDict(extra="forbid")

    job_id: str
    objective: Literal["Econ", "QoL"]
    mode: Literal["single", "multi"]
    project_root: Path
    output_root: Path
    data: DataTrainingConfig
    model: ModelTrainingConfig
    training: TrainLoopConfig

    @training_field_validator("job_id", "objective", "mode", mode="before")
    @classmethod
    def strip_training_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @training_model_validator(mode="after")
    def validate_phase_two_scope(self) -> "TrainingRunConfig":
        if self.mode == "multi":
            if not self.data.components:
                raise ValueError("Multi-head training requires data.components")
        return self
