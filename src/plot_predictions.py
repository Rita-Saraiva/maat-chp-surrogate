"""Plot surrogate predictions against true targets (mean across zones per step).

Reloads a trained checkpoint using the same layout scheme as
``src/eval/cross_dataset.py`` (``family`` / ``mode`` / ``train_dataset`` /
``component`` / ``profile`` / ``source_job_id``), runs it over a dataset split
(default ``val``), and plots predicted vs true targets. Each point is one time step:
the value is averaged across zones. Mode-component targets get a per-mode plot plus a
mode-coloured aggregate; when ``ZONES`` is provided, one coloured series per selected
zone is added (no averaging across the selected zones).

Environment variables mirror ``jobs/evaluate_surrogate.pbs``:
    OBJECTIVE, TRAIN_DATASET, MODE, MODEL, PARAMETERS, SOURCE_JOB_ID (required);
    COMPONENT (required for MODE=single);
    DATASET (dataset to score, default TRAIN_DATASET); SPLIT (default ``val``);
    ZONES (optional region ids); LOADER_MODE, BATCH_SIZE, NUM_WORKERS, MAX_SAMPLES;
    PROJECT_ROOT / DATA_ROOT / OUTPUT_ROOT.

Run: ``python -u -m src.plot_predictions``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config.load import (
    FAMILY_PROFILE_TYPES,
    MULTI_COMPONENT_LABEL,
    TRAINING_OBJECTIVE_COMPONENT_ORDER,
    canonical_family,
    canonical_training_objective,
    require_environment_value,
    resolve_training_job_id,
)
from src.config.schema import ModelTrainingConfig
from src.data.dataset import DatasetManifest
from src.data.graph import load_region_graph
from src.data.training import create_training_dataset, create_training_loader
from src.eval import analysis_common as ac
from src.eval.cross_dataset import EvaluationRunConfig, _checkpoint_path
from src.models.registry import (
    create_multihead_surrogate_model,
    create_surrogate_model,
)


def _int_env(name: str, default: int) -> int:
    text = os.environ.get(name, "").strip()
    return int(text) if text else default


def _build_config(env: dict[str, str]) -> EvaluationRunConfig:
    project_root = Path(env.get("PROJECT_ROOT", "/mnt/project")).expanduser()
    objective = canonical_training_objective(require_environment_value(env, "OBJECTIVE"))
    train_dataset = require_environment_value(env, "TRAIN_DATASET")
    dataset = env.get("DATASET", "").strip() or train_dataset
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
        components = None

    data_root = Path(env.get("DATA_ROOT", str(project_root / "data"))).expanduser()
    output_root = Path(env.get("OUTPUT_ROOT", str(project_root))).expanduser()
    max_samples_text = env.get("MAX_SAMPLES", "").strip()

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
        eval_dataset=dataset,
        eval_dataset_dir=data_root / dataset,
        project_root=project_root,
        output_root=output_root,
        loader_mode=env.get("LOADER_MODE", "chunked").strip().lower(),
        batch_size=_int_env("BATCH_SIZE", 1024),
        num_workers=_int_env("NUM_WORKERS", 0),
        pin_memory=False,
        max_samples=int(max_samples_text) if max_samples_text else None,
    )


def _load_model(config: EvaluationRunConfig, dataset: Any):
    checkpoint_path = _checkpoint_path(config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {checkpoint_path}. Verify family/mode/"
            f"train_dataset/component/profile and SOURCE_JOB_ID={config.source_job_id!r}."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])
    profile_payload = checkpoint["profile"]
    model_config = ModelTrainingConfig(
        family=config.family,
        profile=config.profile,
        parameters=FAMILY_PROFILE_TYPES[config.family].model_validate(profile_payload),
    )

    if config.mode == "multi":
        model = create_multihead_surrogate_model(
            config=model_config,
            input_dim=input_dim,
            head_output_dims=dataset.head_output_dims,
            head_slices=dataset.head_slices,
            graph_search_roots=None,
        )
    else:
        model = create_surrogate_model(
            config=model_config,
            input_dim=input_dim,
            output_dim=output_dim,
            graph_search_roots=None,
        )
    model.load(checkpoint_path)
    return model, input_dim, output_dim, checkpoint_path


def _component_specs(
    config: EvaluationRunConfig,
    dataset: Any,
    manifest: DatasetManifest,
    output_dim: int,
) -> list[tuple[str, tuple[int, int], list[str] | None]]:
    """Return ``(component, (flat_start, flat_stop), modes)`` for each head to plot."""
    modes_by_target = {name: modes for name, _cols, modes, _type in ac.target_layout(manifest.payload)}
    if config.mode == "multi":
        specs: list[tuple[str, tuple[int, int], list[str] | None]] = []
        for name, (start, stop) in dataset.head_slices.items():
            specs.append((name, (start, stop), modes_by_target.get(name)))
        return specs
    return [(config.component, (0, output_dim), modes_by_target.get(config.component))]


def _plot_component(
    split: str,
    component: str,
    pred: np.ndarray,
    true: np.ndarray,
    modes: list[str] | None,
    num_regions: int,
    zones: list[tuple[int, int, str]],
    output_dir: Path,
    max_scatter_points: int,
    rng: np.random.Generator,
) -> None:
    """Reshape a head's flat block to ``[N, R, cols]`` and plot pred vs true."""
    columns = pred.shape[1] // num_regions
    pred_block = pred.reshape(pred.shape[0], num_regions, columns)
    true_block = true.reshape(true.shape[0], num_regions, columns)
    labels = modes if (modes and len(modes) == columns) else [f"col{c}" for c in range(columns)]

    for col, label in enumerate(labels):
        x_true = true_block[:, :, col].mean(axis=1)
        y_pred = pred_block[:, :, col].mean(axis=1)
        ac.save_scatter(
            [(f"{label} (mean over zones)", x_true, y_pred)],
            xlabel=f"true {component}[{label}]",
            ylabel=f"predicted {component}[{label}]",
            title=f"{split}: {component}[{label}] pred vs true (mean over zones)",
            path=output_dir / f"pred_{ac.safe_name(component)}_{ac.safe_name(label)}_meanzones.png",
            max_points=max_scatter_points,
            rng=rng,
            identity_line=True,
        )

    if columns > 1:
        aggregate = [
            (label, true_block[:, :, col].mean(axis=1), pred_block[:, :, col].mean(axis=1))
            for col, label in enumerate(labels)
        ]
        ac.save_scatter(
            aggregate,
            xlabel=f"true {component}",
            ylabel=f"predicted {component}",
            title=f"{split}: {component} pred vs true (mean over zones, all modes)",
            path=output_dir / f"pred_{ac.safe_name(component)}_aggregate_meanzones.png",
            max_points=max_scatter_points,
            rng=rng,
            identity_line=True,
        )

    if not zones:
        return

    for col, label in enumerate(labels):
        series = [
            (f"{zlabel} ({zone_id})", true_block[:, node, col], pred_block[:, node, col])
            for zone_id, node, zlabel in zones
        ]
        ac.save_scatter(
            series,
            xlabel=f"true {component}[{label}]",
            ylabel=f"predicted {component}[{label}]",
            title=f"{split}: {component}[{label}] pred vs true (per zone)",
            path=output_dir / f"pred_{ac.safe_name(component)}_{ac.safe_name(label)}_zones.png",
            max_points=max_scatter_points,
            rng=rng,
            identity_line=True,
        )


def main() -> None:
    env = dict(os.environ)
    config = _build_config(env)
    split = env.get("SPLIT", "val").strip().lower() or "val"
    max_scatter_points = max(0, _int_env("MAX_SCATTER_POINTS", 200_000))
    seed = _int_env("SEED", 42)

    torch.set_num_threads(max(1, _int_env("OMP_NUM_THREADS", 4)))

    manifest = DatasetManifest(config.eval_dataset_dir)
    num_regions = manifest.num_regions

    dataset = create_training_dataset(
        dataset_dir=config.eval_dataset_dir,
        split=split,
        loader_mode=config.loader_mode,
        component=config.component,
        components=config.components,
        max_samples=config.max_samples,
    )
    loader = create_training_loader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    model, input_dim, output_dim, checkpoint_path = _load_model(config, dataset)
    print(f"[plot-pred] checkpoint={checkpoint_path}", flush=True)
    print(
        f"[plot-pred] dataset={config.eval_dataset} split={split} "
        f"rows={len(dataset)} input_dim={input_dim} output_dim={output_dim}",
        flush=True,
    )

    predictions, targets = model.predict(loader)
    pred = predictions.detach().cpu().numpy().astype(np.float64)
    true = targets.detach().cpu().numpy().astype(np.float64)

    zones: list[tuple[int, int, str]] = []
    zone_ids = ac.parse_zone_ids(env.get("ZONES", ""))
    if zone_ids:
        graph = load_region_graph(search_roots=[config.project_root, config.eval_dataset_dir])
        zones = ac.zone_indices(graph.region_ids, graph.names, zone_ids)
        print(f"Selected zones: {[f'{label} ({zid})' for zid, _, label in zones]}", flush=True)

    output_dir = (
        config.output_root
        / "results"
        / "analysis"
        / "predictions"
        / config.family
        / config.mode
        / config.eval_dataset
        / config.component
        / config.profile
        / config.source_job_id
        / split
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for component, (start, stop), modes in _component_specs(config, dataset, manifest, output_dim):
        _plot_component(
            split=split,
            component=component,
            pred=pred[:, start:stop],
            true=true[:, start:stop],
            modes=modes,
            num_regions=num_regions,
            zones=zones,
            output_dir=output_dir,
            max_scatter_points=max_scatter_points,
            rng=rng,
        )

    print(f"\nPrediction plots written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
