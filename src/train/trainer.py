from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config.schema import TrainingRunConfig
from src.data.dataset import DatasetManifest
from src.data.training import (
    create_training_dataset,
    create_training_loader,
    inspect_dataset_dimensions,
)
from src.eval.metrics import add_policy_metrics
from src.models.registry import (
    create_multihead_surrogate_model,
    create_surrogate_model,
)
from src.results.training_results import (
    RunHistoryRecord,
    TrainingResultRecord,
    append_fraction_history,
    append_run_history,
    write_training_result,
)
from src.train.loop import set_random_seed


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoints_dir: Path
    config_dir: Path
    metrics_dir: Path
    history_dir: Path
    model_path: Path
    raw_state_path: Path


def create_run_paths(config: TrainingRunConfig) -> RunPaths:
    run_dir = (
        config.output_root
        / "outputs"
        / config.model.family
        / config.mode
        / config.data.dataset
        / config.data.component
        / config.model.profile
        / config.job_id
    )
    checkpoints_dir = run_dir / "checkpoints"
    config_dir = run_dir / "config"
    metrics_dir = run_dir / "metrics"
    history_dir = run_dir / "history"
    for directory in (run_dir, checkpoints_dir, config_dir, metrics_dir, history_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        config_dir=config_dir,
        metrics_dir=metrics_dir,
        history_dir=history_dir,
        model_path=checkpoints_dir / "model.pt",
        raw_state_path=checkpoints_dir / "best_state.pt",
    )


def write_resolved_config(config: TrainingRunConfig, path: Path) -> None:
    payload = config.model_dump(mode="json")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_surrogate(config: TrainingRunConfig) -> dict[str, Any]:
    paths = create_run_paths(config)
    write_resolved_config(config, paths.config_dir / "resolved_config.yaml")

    is_multi = config.mode == "multi"
    components = config.data.components if is_multi else None

    print(
        f"[training] dataset={config.data.dataset} component={config.data.component} "
        f"mode={config.mode} model={config.model.family} profile={config.model.profile}",
        flush=True,
    )
    if is_multi:
        print(
            f"[training] components={components}",
            flush=True,
        )
    print(
        f"[training] loader_mode={config.data.loader_mode} dataset_dir={config.data.dataset_dir}",
        flush=True,
    )

    # Datasize-robustness sweeps train on the first k whole shards of the TRAIN
    # split; validation/test stay full so the evaluation set is constant across the
    # fraction curve. Shards are fixed-size, so summing the first k shard sizes and
    # capping the train dataset length reproduces "the first k shards" exactly.
    train_fraction = config.data.train_fraction
    train_max_samples = config.data.max_samples
    train_rows_target: int | None = None
    if train_fraction is not None:
        train_shards = DatasetManifest(config.data.dataset_dir).shards("train")
        shard_count = len(train_shards)
        if shard_count == 0:
            raise ValueError("Train split has no shards; cannot apply train_fraction")
        selected = min(shard_count, max(1, round(train_fraction * shard_count)))
        train_rows_target = sum(int(entry["n_samples"]) for entry in train_shards[:selected])
        train_max_samples = (
            train_rows_target
            if train_max_samples is None
            else min(train_max_samples, train_rows_target)
        )
        print(
            f"[training] train_fraction={train_fraction} -> first {selected}/{shard_count} "
            f"train shards ({train_rows_target} rows); val/test stay full",
            flush=True,
        )

    train_dataset = create_training_dataset(
        dataset_dir=config.data.dataset_dir,
        split="train",
        loader_mode=config.data.loader_mode,
        component=config.data.component,
        components=components,
        max_samples=train_max_samples,
    )
    validation_dataset = create_training_dataset(
        dataset_dir=config.data.dataset_dir,
        split="val",
        loader_mode=config.data.loader_mode,
        component=config.data.component,
        components=components,
        max_samples=config.data.max_samples,
    )
    test_dataset = create_training_dataset(
        dataset_dir=config.data.dataset_dir,
        split="test",
        loader_mode=config.data.loader_mode,
        component=config.data.component,
        components=components,
        max_samples=config.data.max_samples,
    )

    input_dim, output_dim = inspect_dataset_dimensions(train_dataset)
    validation_dims = inspect_dataset_dimensions(validation_dataset)
    test_dims = inspect_dataset_dimensions(test_dataset)
    if validation_dims != (input_dim, output_dim) or test_dims != (input_dim, output_dim):
        raise ValueError(
            "Split dimensions do not match: "
            f"train={(input_dim, output_dim)}, val={validation_dims}, test={test_dims}"
        )

    head_output_dims: dict[str, int] | None = None
    head_slices: dict[str, tuple[int, int]] | None = None
    if is_multi:
        head_output_dims = train_dataset.head_output_dims
        head_slices = train_dataset.head_slices
        if not head_output_dims or not head_slices:
            raise ValueError(
                "Multi-head training requires head metadata; the dataset did not "
                "expose head_output_dims / head_slices."
            )
        if train_dataset.head_slices != validation_dataset.head_slices or (
            train_dataset.head_slices != test_dataset.head_slices
        ):
            raise ValueError("Head layouts differ across splits.")

    train_loader = create_training_loader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    validation_loader = create_training_loader(
        dataset=validation_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    test_loader = create_training_loader(
        dataset=test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    set_random_seed(config.training.seed, config.training.deterministic)
    # The GNN family resolves its adjacency files relative to these roots; the
    # other families ignore them.
    graph_search_roots = [config.project_root, config.data.dataset_dir]
    if is_multi:
        assert head_output_dims is not None and head_slices is not None
        model = create_multihead_surrogate_model(
            config=config.model,
            input_dim=input_dim,
            head_output_dims=head_output_dims,
            head_slices=head_slices,
            graph_search_roots=graph_search_roots,
        )
    else:
        model = create_surrogate_model(
            config=config.model,
            input_dim=input_dim,
            output_dim=output_dim,
            graph_search_roots=graph_search_roots,
        )
    
    metadata = {
        "Job Id": config.job_id,
        "Dataset": config.data.dataset,
        "Mode": config.mode,
        "Model": config.model.family,
        "Objective": config.objective,
        "Component": config.data.component,
        "Input Dimension": input_dim,
        "Output Dimension": output_dim,
    }
    if is_multi:
        metadata["Components"] = list(components or [])
        metadata["Heads"] = dict(head_output_dims or {})

    def _periodic_save(epoch: int) -> None:
        model.save(paths.model_path, metadata=metadata)
        print(f"[training] checkpoint saved at epoch {epoch}", flush=True)

    train_start = time.perf_counter()
    training_result = model.fit(
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config.training,
        checkpoint_path=paths.raw_state_path,
        periodic_save_callback=_periodic_save,
    )
    training_time_seconds = time.perf_counter() - train_start
    # Final save uses best weights (restored by fit_torch_model / GP fit after loop).
    model.save(paths.model_path, metadata=metadata)

    validation_metrics = model.evaluate(validation_loader)
    test_metrics = model.evaluate(test_loader)

    # Policy (rank) evaluation. Kendall/Spearman correlations cannot be streamed
    # like the regression accumulators, so materialize predictions/targets via
    # predict() (original target space, all families) and merge the policy metrics
    # into each split's metric dict. For multi-head runs the overall metrics use
    # the full arrays and each head is scored on its contiguous output slice.
    policy_head_slices = head_slices if is_multi else None
    validation_predictions, validation_targets = model.predict(validation_loader)
    add_policy_metrics(
        validation_metrics,
        validation_predictions,
        validation_targets,
        head_slices=policy_head_slices,
    )
    test_predictions, test_targets = model.predict(test_loader)
    add_policy_metrics(
        test_metrics,
        test_predictions,
        test_targets,
        head_slices=policy_head_slices,
    )

    validation_overall = validation_metrics["Overall"] if is_multi else validation_metrics
    test_overall = test_metrics["Overall"] if is_multi else test_metrics

    history_path = paths.history_dir / "history.json"
    write_json({"History": training_result["History"]}, history_path)

    metrics_payload = {
        "Validation": validation_metrics,
        "Test": test_metrics,
        "Training": {
            **{
                key: value
                for key, value in training_result.items()
                if key != "History"
            },
            "Training Time (s)": training_time_seconds,
        },
        "Data": {
            "Train Rows": len(train_dataset),
            "Validation Rows": len(validation_dataset),
            "Test Rows": len(test_dataset),
            "Input Dimension": input_dim,
            "Output Dimension": output_dim,
            "Loader Mode": config.data.loader_mode,
        },
        "Paths": {
            "Run Directory": str(paths.run_dir),
            "Model": str(paths.model_path),
            "History": str(history_path),
        },
    }
    if is_multi:
        metrics_payload["Heads"] = dict(head_output_dims or {})
    metrics_path = paths.metrics_dir / "metrics.json"
    write_json(metrics_payload, metrics_path)

    parameters = {
        "Profile": config.model.profile,
        "Architecture": config.model.parameters.model_dump(mode="python"),
        "Training": config.training.model_dump(mode="python"),
        "Data Loader": {
            "Loader Mode": config.data.loader_mode,
            "Batch Size": config.data.batch_size,
            "Num Workers": config.data.num_workers,
            "Max Samples": config.data.max_samples,
            "Train Fraction": train_fraction,
        },
    }
    if is_multi:
        parameters["Components"] = list(components or [])
        parameters["Heads"] = dict(head_output_dims or {})
    record = TrainingResultRecord(
        **{
            "Job Id": config.job_id,
            "Dataset": config.data.dataset,
            "Mode": config.mode,
            "Model": config.model.family,
            "Parameters": parameters,
            "Objective": config.objective,
            "Component": config.data.component,
            "Results": metrics_payload,
        }
    )
    result_paths = write_training_result(
        record=record,
        results_root=config.output_root / "results" / "runs",
        write_yaml=False,
    )

    history_records = _build_history_records(
        config=config,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        is_multi=is_multi,
        training_time_seconds=training_time_seconds,
    )
    history_store = append_run_history(
        records=history_records,
        history_root=config.output_root / "results",
        shard_name=config.job_id,
    )

    # Datasize-robustness runs additionally populate a dedicated ``run_fraction``
    # table (fraction recorded as its own column) so the sweep can be plotted
    # directly without disturbing the main run-history store.
    fraction_store: dict[str, Any] | None = None
    if train_fraction is not None:
        fraction_store = append_fraction_history(
            records=history_records,
            fraction=train_fraction,
            history_root=config.output_root / "results",
            shard_name=config.job_id,
        )

    print("\n" + "=" * 72, flush=True)
    print(
        f"{'MULTI-HEAD' if is_multi else 'SINGLE'} SURROGATE TRAINING COMPLETE",
        flush=True,
    )
    print("=" * 72, flush=True)
    print(f"Run directory:   {paths.run_dir}", flush=True)
    print(f"Input dimension: {input_dim}", flush=True)
    print(f"Output dimension:{output_dim}", flush=True)
    print(f"Best epoch:      {training_result['Best Epoch']}", flush=True)
    print(f"Training time:   {training_time_seconds:.2f} s", flush=True)
    print(f"Validation MSE:  {validation_overall['MSE']:.8f}", flush=True)
    print(f"Validation R2:   {validation_overall['R2']:.8f}", flush=True)
    print(f"Test MSE:        {test_overall['MSE']:.8f}", flush=True)
    print(f"Test R2:         {test_overall['R2']:.8f}", flush=True)
    if is_multi:
        for name in (head_output_dims or {}):
            head_val = validation_metrics["Per Head"][name]
            head_test = test_metrics["Per Head"][name]
            print(
                f"  head {name:<14} val R2={head_val['R2']:.6f} "
                f"test R2={head_test['R2']:.6f}",
                flush=True,
            )
    print(f"History table:   {history_store['table']}", flush=True)
    if fraction_store is not None:
        print(f"Fraction table:  {fraction_store['table_fraction']}", flush=True)
    print("=" * 72, flush=True)

    return {
        "run_paths": paths,
        "training": training_result,
        "validation": validation_metrics,
        "test": test_metrics,
        "result_paths": result_paths,
        "history_paths": history_store,
    }


def _build_history_records(
    config: TrainingRunConfig,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    is_multi: bool,
    training_time_seconds: float,
) -> list[RunHistoryRecord]:
    """Build one aggregated run-history record per component.

    Single-head runs yield a single record for their component; multi-head runs
    yield one record per head so both modes share the same per-component schema.
    ``Validation``, ``Test`` and ``Training Time (s)`` are separate columns.
    """
    base = {
        "Job Id": config.job_id,
        "Dataset": config.data.dataset,
        "Model": config.model.family,
        "Parameters": {"Profile": config.model.profile},
        "Objective": config.objective,
    }
    if not is_multi:
        return [
            RunHistoryRecord(
                **base,
                **{
                    "Mode": config.mode,
                    "Component": config.data.component,
                    "Validation": validation_metrics,
                    "Test": test_metrics,
                    "Training Time (s)": training_time_seconds,
                },
            )
        ]

    records: list[RunHistoryRecord] = []
    for name in validation_metrics["Per Head"]:
        records.append(
            RunHistoryRecord(
                **base,
                **{
                    "Mode": config.mode,
                    "Component": name,
                    "Validation": validation_metrics["Per Head"][name],
                    "Test": test_metrics["Per Head"][name],
                    "Training Time (s)": training_time_seconds,
                },
            )
        )
    return records