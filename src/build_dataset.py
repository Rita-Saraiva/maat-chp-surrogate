#!/usr/bin/env python3
from __future__ import annotations

import csv
import gc
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config.schema import BufferAssignment, DatasetBuildRequest
from src.data.assembly import load_buffer, load_yaml, observation_to_xy, segment_episodes
from src.data.contract import ObservationContract, derive_contract
from src.data.sharding import NpyShardWriter



def resolve_episode_len(request: DatasetBuildRequest) -> int:
    if request.build.episode_len > 0:
        return request.build.episode_len
    first_config = load_yaml(request.sources[0].rl_config)
    return int(first_config.get("env", {}).get("episode_time_steps", 77))


def prepare_staging_directory(request: DatasetBuildRequest) -> Path:
    output_dir = request.output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    safe_job_id = "".join(character if character.isalnum() else "_" for character in request.job_id)
    staging = output_dir.parent / f".{output_dir.name}.building_{safe_job_id or 'local'}"
    if staging.exists():
        if request.build.overwrite:
            shutil.rmtree(staging)
        else:
            raise FileExistsError(
                f"Staging directory already exists: {staging}. "
                "Use build.overwrite=true after verifying that no active job owns it."
            )
    if output_dir.exists() and not request.build.overwrite:
        raise FileExistsError(
            f"Dataset output already exists: {output_dir}. Set build.overwrite=true to replace it."
        )
    staging.mkdir(parents=True)
    return staging


def write_source_tables(
    staging: Path,
    request: DatasetBuildRequest,
    assignments: list[BufferAssignment],
) -> None:
    with (staging / "sources.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["source_index", "raw_dir", "buffer_name", "rl_config"],
        )
        writer.writeheader()
        for source in request.sources:
            writer.writerow(
                {
                    "source_index": source.source_index,
                    "raw_dir": source.raw_dir,
                    "buffer_name": source.buffer_name,
                    "rl_config": source.rl_config,
                }
            )

    with (staging / "source_map.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=[
                "dataset_env_id",
                "split",
                "source_index",
                "source_env_id",
                "source_file",
                "raw_dir",
                "buffer_name",
                "rl_config",
            ],
        )
        writer.writeheader()
        for item in assignments:
            writer.writerow(
                {
                    "dataset_env_id": item.dataset_env_id,
                    "split": item.split,
                    "source_index": item.source_index,
                    "source_env_id": item.source_env_id,
                    "source_file": item.source_file.name,
                    "raw_dir": item.raw_dir,
                    "buffer_name": item.buffer_name,
                    "rl_config": item.rl_config,
                }
            )


def build_split(
    *,
    split: str,
    split_assignments: list[BufferAssignment],
    staging: Path,
    request: DatasetBuildRequest,
    contract: ObservationContract,
    episode_len: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    writer = NpyShardWriter(
        split=split,
        split_dir=staging / "splits" / split,
        samples_per_shard=request.build.samples_per_shard,
        x_shape=(contract.num_regions, contract.x_node_width),
        y_shape=(contract.num_regions, contract.y_node_width),
    )
    buffer_rows: list[dict[str, Any]] = []

    try:
        for assignment in split_assignments:
            observation, action, reward, done = load_buffer(assignment.source_file)
            if observation.ndim != 2 or observation.shape[1] != contract.expected_obs_dim:
                raise ValueError(
                    f"{assignment.source_file.name}: observation shape={observation.shape}, "
                    f"expected=(*, {contract.expected_obs_dim})."
                )
            if action.ndim != 2 or action.shape[1] != contract.num_regions:
                raise ValueError(
                    f"{assignment.source_file.name}: action shape={action.shape}, "
                    f"expected=(*, {contract.num_regions})."
                )

            episodes = segment_episodes(
                done,
                episode_len=episode_len,
                episode_stride=request.build.episode_stride,
                max_episodes=request.build.max_episodes_per_env,
                file_name=assignment.source_file.name,
            )
            rows_kept = 0
            for local_episode_id, (start, stop) in enumerate(episodes):
                for episode_timestep, timestep in enumerate(range(start, stop)):
                    x, y = observation_to_xy(
                        observation[timestep],
                        action[timestep],
                        contract,
                    )
                    writer.add(
                        x,
                        y,
                        {
                            "dataset_env_id": assignment.dataset_env_id,
                            "source_index": assignment.source_index,
                            "source_env_id": assignment.source_env_id,
                            "timestep": timestep,
                            "episode_id": local_episode_id,
                            "episode_timestep": episode_timestep,
                            "reward": float(reward[timestep]),
                            "done": bool(done[timestep]),
                        },
                    )
                    rows_kept += 1

            raw_steps = int(len(observation))
            first_kept = episodes[0][0] if episodes else 0
            last_kept = episodes[-1][1] if episodes else 0
            buffer_rows.append(
                {
                    "split": split,
                    "dataset_env_id": assignment.dataset_env_id,
                    "source_index": assignment.source_index,
                    "source_env_id": assignment.source_env_id,
                    "source_file": assignment.source_file.name,
                    "raw_steps": raw_steps,
                    "episodes_kept": len(episodes),
                    "rows_kept": rows_kept,
                    "reset_rows_dropped": int(first_kept),
                    "partial_tail_dropped": int(raw_steps - last_kept),
                    "done_true_count_total": int(done.sum()),
                }
            )
            del observation, action, reward, done
            gc.collect()
    except Exception:
        writer.abort()
        raise

    return writer.close(), buffer_rows


def write_summary(staging: Path, manifest: dict[str, Any]) -> None:
    dataset = manifest["dataset"]
    storage = manifest["storage"]
    lines = [
        "Environment-level dataset build complete",
        "",
        f"created_at_utc: {manifest['created_at_utc']}",
        f"job_id: {dataset['job_id']}",
        f"dataset: {dataset['name']}",
        f"objective: {dataset['objective']}",
        f"num_sources: {dataset['num_sources']}",
        f"episode_len: {dataset['episode_len']}",
        f"storage_format: {storage['format']}",
        f"samples_per_shard: {storage['samples_per_shard']}",
        f"x_sample_shape: {storage['x_sample_shape']}",
        f"y_sample_shape: {storage['y_sample_shape']}",
        "",
        "Splits:",
    ]
    for split in ("train", "val", "test"):
        split_info = manifest["splits"][split]
        lines.append(
            f"  {split}: environments={split_info['n_environments']} "
            f"samples={split_info['n_samples']} shards={split_info['n_shards']}"
        )
    lines.extend(["", "Targets:"])
    for target, spec in manifest["contract"]["targets"].items():
        lines.append(
            f"  {target}: columns={spec['y_columns']} type={spec['type']} "
            f"input_style={spec['input_style']}"
        )
    lines.extend(["", "Buffers:"])
    for row in manifest["buffer_rows"]:
        lines.append(
            f"  {row['split']} dataset_env_id={row['dataset_env_id']} "
            f"source={row['source_index']} source_env_id={row['source_env_id']} "
            f"file={row['source_file']} raw_steps={row['raw_steps']} "
            f"episodes_kept={row['episodes_kept']} rows_kept={row['rows_kept']} "
            f"reset_rows_dropped={row['reset_rows_dropped']} "
            f"partial_tail_dropped={row['partial_tail_dropped']}"
        )
    (staging / "split_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(
    request: DatasetBuildRequest,
    assignments: list[BufferAssignment],
    contract: ObservationContract | None = None,
) -> Path:
    """Build a complete dataset and atomically publish it to ``request.output_dir``."""
    if not assignments:
        raise ValueError("No buffer assignments were supplied.")

    staging = prepare_staging_directory(request)
    print(f"Building in staging directory: {staging}", flush=True)
    try:
        first_rl_config = load_yaml(request.sources[0].rl_config)
        resolved_contract = contract or derive_contract(
            first_rl_config,
            num_regions=request.build.num_regions,
            action_classes=request.build.action_classes,
            include_action_onehot=request.build.include_action_onehot,
        )
        episode_len = resolve_episode_len(request)
        write_source_tables(staging, request, assignments)

        split_results: dict[str, dict[str, Any]] = {}
        all_rows: list[dict[str, Any]] = []
        for split in ("train", "val", "test"):
            selected = sorted(
                (item for item in assignments if item.split == split),
                key=lambda item: item.dataset_env_id,
            )
            result, rows = build_split(
                split=split,
                split_assignments=selected,
                staging=staging,
                request=request,
                contract=resolved_contract,
                episode_len=episode_len,
            )
            split_results[split] = result
            all_rows.extend(rows)

        manifest: dict[str, Any] = {
            "manifest_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "name": request.folder_name,
                "objective": request.objective,
                "job_id": request.job_id,
                "num_sources": len(request.sources),
                "split_unit": "whole_environment_buffer",
                "segmentation": "done_anchored",
                "episode_len": episode_len,
                "episode_stride": request.build.episode_stride,
                "max_episodes_per_env": request.build.max_episodes_per_env,
                "units": "raw_physical_unscaled",
            },
            "storage": {
                "format": "npy_memmap_shards",
                "default_mode": "chunked",
                "samples_per_shard": request.build.samples_per_shard,
                "x_dtype": "float32",
                "y_dtype": "float32",
                "x_sample_shape": [resolved_contract.num_regions, resolved_contract.x_node_width],
                "y_sample_shape": [resolved_contract.num_regions, resolved_contract.y_node_width],
                "metadata_dtype": "structured_npy",
            },
            "split_weights": request.split.model_dump(),
            "contract": resolved_contract.model_dump(),
            "sources": [source.model_dump(mode="json") for source in request.sources],
            "splits": {
                split: {
                    "n_environments": sum(1 for item in assignments if item.split == split),
                    "n_samples": split_results[split]["n_samples"],
                    "n_shards": split_results[split]["n_shards"],
                    "shards": split_results[split]["shards"],
                }
                for split in ("train", "val", "test")
            },
            "buffer_rows": all_rows,
        }
        with (staging / "manifest.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False)
        write_summary(staging, manifest)

        if request.output_dir.exists():
            shutil.rmtree(request.output_dir)
        os.replace(staging, request.output_dir)
        print(f"Published dataset: {request.output_dir}", flush=True)
        return request.output_dir
    except Exception:
        print(f"Build failed; staging directory retained for inspection: {staging}", flush=True)
        raise
