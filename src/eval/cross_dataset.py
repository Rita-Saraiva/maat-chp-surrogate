from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.config.schema import ModelTrainingConfig
from src.data.training import (
    create_training_dataset,
    create_training_loader,
    inspect_dataset_dimensions,
)
from src.models.registry import (
    create_multihead_surrogate_model,
    create_surrogate_model,
)
from src.results.training_results import (
    CrossEvalRecord,
    append_cross_eval_history,
)


@dataclass(frozen=True)
class EvaluationRunConfig:
    """Locate a trained checkpoint and evaluate it on another dataset's test split.

    The checkpoint is identified by the same fields that ``create_run_paths`` uses
    to lay out a training run (``family`` / ``mode`` / ``train_dataset`` /
    ``component`` / ``profile`` / ``source_job_id``). ``job_id`` is the id of *this*
    evaluation run (used for shard naming and artifacts), while ``source_job_id`` is
    the id of the training run that produced the checkpoint.
    """

    job_id: str
    objective: str
    mode: str
    family: str
    profile: str
    component: str
    components: list[str] | None
    source_job_id: str
    train_dataset: str
    eval_dataset: str
    eval_dataset_dir: Path
    project_root: Path
    output_root: Path
    loader_mode: str = "chunked"
    batch_size: int = 1024
    num_workers: int = 0
    pin_memory: bool = True
    max_samples: int | None = None


def _checkpoint_path(config: EvaluationRunConfig) -> Path:
    return (
        config.output_root
        / "outputs"
        / config.family
        / config.mode
        / config.train_dataset
        / config.component
        / config.profile
        / config.source_job_id
        / "checkpoints"
        / "model.pt"
    )


def _eval_run_dir(config: EvaluationRunConfig) -> Path:
    return (
        config.output_root
        / "eval_outputs"
        / config.family
        / config.mode
        / config.train_dataset
        / config.eval_dataset
        / config.component
        / config.profile
        / config.source_job_id
        / config.job_id
    )


def evaluate_cross_dataset(config: EvaluationRunConfig) -> dict[str, Any]:
    """Reload a trained checkpoint and score it on ``eval_dataset``'s test split."""
    from src.config.load import FAMILY_PROFILE_TYPES

    is_multi = config.mode == "multi"

    checkpoint_path = _checkpoint_path(config)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained checkpoint not found: {checkpoint_path}. Verify the model "
            f"fields (family={config.family!r}, mode={config.mode!r}, "
            f"train_dataset={config.train_dataset!r}, component={config.component!r}, "
            f"profile={config.profile!r}) and source_job_id={config.source_job_id!r}."
        )

    print(
        f"[cross-eval] source_job={config.source_job_id} "
        f"train_dataset={config.train_dataset} eval_dataset={config.eval_dataset} "
        f"family={config.family} mode={config.mode} component={config.component} "
        f"profile={config.profile}",
        flush=True,
    )
    print(f"[cross-eval] checkpoint={checkpoint_path}", flush=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_family = str(checkpoint.get("family", "")).strip()
    if checkpoint_family and checkpoint_family != config.family:
        raise ValueError(
            f"Checkpoint family={checkpoint_family!r} does not match requested "
            f"family={config.family!r}."
        )
    input_dim = int(checkpoint["input_dim"])
    output_dim = int(checkpoint["output_dim"])
    profile_payload = checkpoint.get("profile")
    if not isinstance(profile_payload, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not carry a model profile; "
            "cannot rebuild the surrogate for evaluation."
        )
    profile_type = FAMILY_PROFILE_TYPES[config.family]
    model_config = ModelTrainingConfig(
        family=config.family,
        profile=config.profile,
        parameters=profile_type.model_validate(profile_payload),
    )

    test_dataset = create_training_dataset(
        dataset_dir=config.eval_dataset_dir,
        split="test",
        loader_mode=config.loader_mode,
        component=config.component,
        components=config.components,
        max_samples=config.max_samples,
    )

    eval_input_dim, eval_output_dim = inspect_dataset_dimensions(test_dataset)
    if (eval_input_dim, eval_output_dim) != (input_dim, output_dim):
        raise ValueError(
            "Evaluation dataset schema does not match the trained checkpoint: "
            f"checkpoint=({input_dim}, {output_dim}), "
            f"eval_dataset {config.eval_dataset}=({eval_input_dim}, {eval_output_dim}). "
            "Cross-dataset evaluation requires identical input/output dimensions."
        )

    head_output_dims: dict[str, int] | None = None
    head_slices: dict[str, tuple[int, int]] | None = None
    if is_multi:
        head_output_dims = test_dataset.head_output_dims
        head_slices = test_dataset.head_slices
        if not head_output_dims or not head_slices:
            raise ValueError(
                "Multi-head evaluation requires head metadata; the evaluation "
                "dataset did not expose head_output_dims / head_slices."
            )

    test_loader = create_training_loader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    # The GNN family rebuilds its adjacency from the checkpoint on load, so no graph
    # search roots are needed here; the other families ignore this argument.
    if is_multi:
        assert head_output_dims is not None and head_slices is not None
        model = create_multihead_surrogate_model(
            config=model_config,
            input_dim=input_dim,
            head_output_dims=head_output_dims,
            head_slices=head_slices,
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

    test_metrics = model.evaluate(test_loader)
    test_overall = test_metrics["Overall"] if is_multi else test_metrics

    run_dir = _eval_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "Test": test_metrics,
        "Source": {
            "Source Job Id": config.source_job_id,
            "Train Dataset": config.train_dataset,
            "Checkpoint": str(checkpoint_path),
        },
        "Eval": {
            "Dataset": config.eval_dataset,
            "Objective": config.objective,
            "Mode": config.mode,
            "Component": config.component,
            "Model": config.family,
            "Profile": config.profile,
            "Test Rows": len(test_dataset),
            "Input Dimension": input_dim,
            "Output Dimension": output_dim,
            "Loader Mode": config.loader_mode,
        },
    }
    if is_multi:
        metrics_payload["Heads"] = dict(head_output_dims or {})
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, default=str)

    records = _build_cross_eval_records(
        config=config,
        test_metrics=test_metrics,
        is_multi=is_multi,
    )
    history_store = append_cross_eval_history(
        records=records,
        history_root=config.output_root / "results",
        shard_name=config.job_id,
    )

    print("\n" + "=" * 72, flush=True)
    print("CROSS-DATASET EVALUATION COMPLETE", flush=True)
    print("=" * 72, flush=True)
    print(f"Eval run directory: {run_dir}", flush=True)
    print(f"Source job id:      {config.source_job_id}", flush=True)
    print(f"Train dataset:      {config.train_dataset}", flush=True)
    print(f"Eval dataset:       {config.eval_dataset}", flush=True)
    print(f"Test rows:          {len(test_dataset)}", flush=True)
    print(f"Test MSE:           {test_overall['MSE']:.8f}", flush=True)
    print(f"Test R2:            {test_overall['R2']:.8f}", flush=True)
    if is_multi:
        for name in (head_output_dims or {}):
            head_test = test_metrics["Per Head"][name]
            print(
                f"  head {name:<14} test R2={head_test['R2']:.6f}",
                flush=True,
            )
    print(f"Cross-eval table:   {history_store['table_cross_eval']}", flush=True)
    print("=" * 72, flush=True)

    return {
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "test": test_metrics,
        "metrics_path": metrics_path,
        "history_paths": history_store,
    }


def _build_cross_eval_records(
    config: EvaluationRunConfig,
    test_metrics: dict[str, Any],
    is_multi: bool,
) -> list[CrossEvalRecord]:
    """Build one cross-eval record per component (one per head for multi)."""
    base = {
        "Job Id": config.job_id,
        "Dataset": config.eval_dataset,
        "Model": config.family,
        "Parameters": {"Profile": config.profile},
        "Objective": config.objective,
        "Train Dataset": config.train_dataset,
        "Source Job Id": config.source_job_id,
    }
    if not is_multi:
        return [
            CrossEvalRecord(
                **base,
                **{
                    "Mode": config.mode,
                    "Component": config.component,
                    "Test": test_metrics,
                },
            )
        ]

    records: list[CrossEvalRecord] = []
    for name in test_metrics["Per Head"]:
        records.append(
            CrossEvalRecord(
                **base,
                **{
                    "Mode": config.mode,
                    "Component": name,
                    "Test": test_metrics["Per Head"][name],
                },
            )
        )
    return records
