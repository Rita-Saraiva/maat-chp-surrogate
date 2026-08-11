from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from src.config.schema import BuildSettings, DataConfig, RunConfig, SourceRoots


ModelT = TypeVar("ModelT", bound=BaseModel)


def load_yaml_mapping(path: str | Path) -> dict:
    yaml_path = Path(path).expanduser()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {yaml_path}")
    return payload


def load_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    return model_type.model_validate(load_yaml_mapping(path))


def load_build_settings(path: str | Path) -> BuildSettings:
    return load_model(path, BuildSettings)


def load_source_roots(path: str | Path) -> SourceRoots | None:
    yaml_path = Path(path).expanduser()
    if not yaml_path.is_file():
        return None
    return load_model(yaml_path, SourceRoots)


def load_data_config(path: str | Path) -> DataConfig:
    return load_model(path, DataConfig)


def load_run_config(path: str | Path) -> RunConfig:
    return load_model(path, RunConfig)


# Phase 2 training configuration loading
import os as training_os
from pathlib import Path as TrainingPath
from typing import Any as TrainingAny
from typing import Mapping as TrainingMapping

import yaml as training_yaml

from src.config.schema import (
    AutoProfileConfig,
    DataTrainingConfig,
    GNNProfileConfig,
    GPProfileConfig,
    MLPProfileConfig,
    ModelTrainingConfig,
    TrainLoopConfig,
    TrainingRunConfig,
    VAEProfileConfig,
    XGBProfileConfig,
)


FAMILY_ALIASES = {
    "MLP": "MLP",
    "AUTO": "Auto",
    "VAE": "VAE",
    "GP": "GP",
    "XGB": "XGB",
    "XGBOOST": "XGB",
    "GNN": "GNN",
}

FAMILY_PROFILE_TYPES = {
    "MLP": MLPProfileConfig,
    "Auto": AutoProfileConfig,
    "VAE": VAEProfileConfig,
    "GP": GPProfileConfig,
    "XGB": XGBProfileConfig,
    "GNN": GNNProfileConfig,
}


def canonical_family(value: str) -> str:
    key = value.strip().upper()
    if key not in FAMILY_ALIASES:
        raise ValueError(
            f"Unknown model family {value!r}; expected one of {sorted(FAMILY_ALIASES.values())}"
        )
    return FAMILY_ALIASES[key]


TRAINING_OBJECTIVE_ALIASES = {
    "econ": "Econ",
    "economic": "Econ",
    "qol": "QoL",
    "qualityoflife": "QoL",
    "quality_of_life": "QoL",
}

TRAINING_OBJECTIVE_COMPONENTS = {
    "Econ": {"delay", "cancel", "infra"},
    "QoL": {"qol", "action", "maintenance"},
}

# Deterministic head order for multi-head runs. Each head reproduces the exact
# single-head output for its component, so the order fixes the concatenation
# layout used by head slices, checkpoints, and the results store.
TRAINING_OBJECTIVE_COMPONENT_ORDER = {
    "Econ": ("delay", "cancel", "infra"),
    "QoL": ("qol", "action", "maintenance"),
}

MULTI_COMPONENT_LABEL = "multi_all"


def read_yaml_mapping(path: str | TrainingPath) -> dict[str, TrainingAny]:
    yaml_path = TrainingPath(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = training_yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {yaml_path}")
    return payload


def canonical_training_objective(value: str) -> str:
    text = value.strip()
    if text in TRAINING_OBJECTIVE_COMPONENTS:
        return text
    key = text.lower().replace(" ", "").replace("-", "_")
    if key not in TRAINING_OBJECTIVE_ALIASES:
        raise ValueError(f"Unknown objective {value!r}; expected Econ or QoL")
    return TRAINING_OBJECTIVE_ALIASES[key]


def require_environment_value(environment: TrainingMapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise ValueError(f"Required environment variable {name} is missing or empty")
    return value


def resolve_training_job_id(environment: TrainingMapping[str, str]) -> str:
    return (
        environment.get("PBS_JOBID", "").strip()
        or environment.get("JOB_ID", "").strip()
        or "local"
    )


def load_model_profile(path: TrainingPath, family: str, profile_name: str):
    payload = read_yaml_mapping(path)
    configured_family = str(payload.get("family", "")).strip()
    if configured_family and configured_family != family:
        raise ValueError(
            f"Model config family={configured_family!r}, requested family={family!r}"
        )
    profiles = payload.get("profiles", {})
    profile = profiles.get(profile_name) if isinstance(profiles, dict) else None
    if not isinstance(profile, dict):
        available = (
            ", ".join(sorted(str(name) for name in profiles))
            if isinstance(profiles, dict)
            else ""
        )
        raise ValueError(
            f"Unknown model profile {profile_name!r}. Available profiles: {available}"
        )
    profile_type = FAMILY_PROFILE_TYPES[family]
    return profile_type.model_validate(profile)


def load_training_run(
    environment: TrainingMapping[str, str] | None = None,
) -> TrainingRunConfig:
    env = dict(training_os.environ if environment is None else environment)
    project_root = TrainingPath(env.get("PROJECT_ROOT", "/mnt/project")).expanduser()

    objective = canonical_training_objective(require_environment_value(env, "OBJECTIVE"))
    dataset = require_environment_value(env, "DATASET")
    mode = require_environment_value(env, "MODE").lower()
    family = canonical_family(require_environment_value(env, "MODEL"))
    profile_name = require_environment_value(env, "PARAMETERS")

    if mode not in ("single", "multi"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'single' or 'multi'")

    if mode == "multi":
        components: list[str] | None = list(TRAINING_OBJECTIVE_COMPONENT_ORDER[objective])
        component = MULTI_COMPONENT_LABEL
    else:
        component = require_environment_value(env, "COMPONENT").lower()
        allowed_components = TRAINING_OBJECTIVE_COMPONENTS[objective]
        if component not in allowed_components:
            raise ValueError(
                f"Component {component!r} does not belong to objective {objective!r}. "
                f"Allowed components: {sorted(allowed_components)}"
            )
        components = None

    model_config_path = TrainingPath(
        env.get(
            "MODEL_CONFIG",
            str(project_root / "configs" / "model" / f"{family.lower()}.yaml"),
        )
    )
    train_config_path = TrainingPath(
        env.get("TRAIN_CONFIG", str(project_root / "configs" / "train" / "default.yaml"))
    )

    model_parameters = load_model_profile(model_config_path, family, profile_name)
    training_payload = read_yaml_mapping(train_config_path)
    training = TrainLoopConfig.model_validate(
        training_payload.get("training", training_payload)
    )

    batch_size = int(env.get("BATCH_SIZE", training_payload.get("batch_size", 1024)))
    num_workers = int(env.get("NUM_WORKERS", training_payload.get("num_workers", 1)))
    loader_mode = env.get(
        "LOADER_MODE", str(training_payload.get("loader_mode", "chunked"))
    ).strip().lower()
    max_samples_text = env.get("MAX_SAMPLES", "").strip()
    max_samples = int(max_samples_text) if max_samples_text else None

    train_fraction_text = env.get("TRAIN_FRACTION", "").strip()
    train_fraction = float(train_fraction_text) if train_fraction_text else None

    data_root = TrainingPath(env.get("DATA_ROOT", str(project_root / "data")))
    dataset_dir = data_root / dataset
    output_root = TrainingPath(env.get("OUTPUT_ROOT", str(project_root))).expanduser()

    return TrainingRunConfig(
        job_id=resolve_training_job_id(env),
        objective=objective,
        mode=mode,
        project_root=project_root,
        output_root=output_root,
        data=DataTrainingConfig(
            dataset=dataset,
            dataset_dir=dataset_dir,
            loader_mode=loader_mode,
            component=component,
            components=components,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=bool(training_payload.get("pin_memory", True)),
            max_samples=max_samples,
            train_fraction=train_fraction,
        ),
        model=ModelTrainingConfig(
            family=family,
            profile=profile_name,
            parameters=model_parameters,
        ),
        training=training,
    )


def load_evaluation_run(
    environment: TrainingMapping[str, str] | None = None,
):
    """Build a cross-dataset evaluation config from environment variables.

    Locates a trained checkpoint by the same fields used to lay out a training run
    (``MODEL`` / ``MODE`` / ``TRAIN_DATASET`` / ``COMPONENT`` / ``PARAMETERS`` /
    ``SOURCE_JOB_ID``) and evaluates it on the test split of ``EVAL_DATASET``. The
    evaluation run's own id comes from ``PBS_JOBID`` / ``JOB_ID`` (used for shard
    naming and artifacts), independent of the trained run's ``SOURCE_JOB_ID``.
    """
    from src.eval.cross_dataset import EvaluationRunConfig

    env = dict(training_os.environ if environment is None else environment)
    project_root = TrainingPath(env.get("PROJECT_ROOT", "/mnt/project")).expanduser()

    objective = canonical_training_objective(require_environment_value(env, "OBJECTIVE"))
    train_dataset = require_environment_value(env, "TRAIN_DATASET")
    eval_dataset = require_environment_value(env, "EVAL_DATASET")
    mode = require_environment_value(env, "MODE").lower()
    family = canonical_family(require_environment_value(env, "MODEL"))
    profile_name = require_environment_value(env, "PARAMETERS")
    source_job_id = require_environment_value(env, "SOURCE_JOB_ID")

    if mode not in ("single", "multi"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'single' or 'multi'")

    if mode == "multi":
        components: list[str] | None = list(TRAINING_OBJECTIVE_COMPONENT_ORDER[objective])
        component = MULTI_COMPONENT_LABEL
    else:
        component = require_environment_value(env, "COMPONENT").lower()
        allowed_components = TRAINING_OBJECTIVE_COMPONENTS[objective]
        if component not in allowed_components:
            raise ValueError(
                f"Component {component!r} does not belong to objective {objective!r}. "
                f"Allowed components: {sorted(allowed_components)}"
            )
        components = None

    loader_mode = env.get("LOADER_MODE", "chunked").strip().lower()
    batch_size = int(env.get("BATCH_SIZE", 1024))
    num_workers = int(env.get("NUM_WORKERS", 0))
    max_samples_text = env.get("MAX_SAMPLES", "").strip()
    max_samples = int(max_samples_text) if max_samples_text else None

    data_root = TrainingPath(env.get("DATA_ROOT", str(project_root / "data")))
    eval_dataset_dir = data_root / eval_dataset
    output_root = TrainingPath(env.get("OUTPUT_ROOT", str(project_root))).expanduser()

    return EvaluationRunConfig(
        job_id=resolve_training_job_id(env),
        objective=objective,
        mode=mode,
        family=family,
        profile=profile_name,
        component=component,
        components=components,
        source_job_id=source_job_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_dataset_dir=eval_dataset_dir,
        project_root=project_root,
        output_root=output_root,
        loader_mode=loader_mode,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=bool(int(env.get("PIN_MEMORY", "1"))),
        max_samples=max_samples,
    )
