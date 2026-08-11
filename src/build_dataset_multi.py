#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import os
import random
import re
from pathlib import Path
from typing import Iterable

from src.build_dataset import build_dataset
from src.config.load import load_build_settings, load_source_roots
from src.config.schema import (
    BufferAssignment,
    DatasetBuildRequest,
    SourceConfig,
)
from src.data.assembly import env_id_from_name, load_yaml
from src.data.contract import ObservationContract, derive_contract


def csv_values(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def broadcast(values: list[str], size: int, name: str) -> list[str]:
    if len(values) == size:
        return values
    if len(values) == 1:
        return values * size
    raise ValueError(f"{name} must contain either 1 or {size} comma-separated values; got {len(values)}")


def natural_key(path: Path) -> list[object]:
    return [int(item) if item.isdigit() else item for item in re.split(r"(\d+)", path.name)]


def resolve_rl_config(raw_dir: Path, config_name: str) -> Path:
    supplied = Path(config_name)
    if supplied.is_absolute():
        candidate = supplied
    elif supplied.suffix.lower() in {".yaml", ".yml"}:
        candidate = raw_dir / "configs" / supplied.name
    else:
        candidate = raw_dir / "configs" / f"{config_name}.yaml"

    if candidate.is_file():
        return candidate

    config_dir = raw_dir / "configs"
    available = [item.name for item in config_dir.glob("*.y*ml")] if config_dir.is_dir() else []
    suggestions = difflib.get_close_matches(candidate.name, available, n=3, cutoff=0.5)
    hint = f" Closest files: {suggestions}" if suggestions else ""
    raise FileNotFoundError(f"RL config does not exist: {candidate}.{hint}")


def find_source_files(raw_dir: Path, buffer_name: str) -> list[Path]:
    base = raw_dir / "cache" if (raw_dir / "cache").is_dir() else raw_dir
    files = sorted(base.glob(f"env_buffer_{buffer_name}_*.pkl"), key=natural_key)
    if not files:
        raise FileNotFoundError(
            f"No buffers matched {base / f'env_buffer_{buffer_name}_*.pkl'}"
        )
    # Repeated source environment ids are allowed. This occurs when one broad
    # BUFFER_NAME matches several climate scenarios; dataset_env_id remains
    # globally unique and source_file preserves the original identity.
    return files


def parse_env_ids(text: str) -> set[int] | None:
    values = csv_values(text)
    return {int(value) for value in values} if values else None


def weighted_split_names(
    count: int,
    *,
    train_weight: int,
    val_weight: int,
    test_weight: int,
) -> list[str]:
    weights = {"train": train_weight, "val": val_weight, "test": test_weight}
    total_weight = sum(weights.values())
    assigned = {name: 0 for name in weights}
    names: list[str] = []

    # Match the existing behavior: validation wins ties, then test, then train.
    tie_order = {"train": 0, "test": 1, "val": 2}
    for next_total in range(1, count + 1):
        scores = {
            name: weight * next_total - assigned[name] * total_weight
            for name, weight in weights.items()
        }
        selected = max(scores, key=lambda name: (scores[name], tie_order[name]))
        assigned[selected] += 1
        names.append(selected)
    return names


def validate_contracts(
    sources: list[SourceConfig],
    *,
    num_regions: int,
    action_classes: int,
    include_action_onehot: bool,
    requested_episode_len: int,
) -> tuple[ObservationContract, int]:
    contracts: list[tuple[Path, ObservationContract, int]] = []
    for source in sources:
        rl_config = load_yaml(source.rl_config)
        contract = derive_contract(
            rl_config,
            num_regions=num_regions,
            action_classes=action_classes,
            include_action_onehot=include_action_onehot,
        )
        yaml_episode_len = int(rl_config.get("env", {}).get("episode_time_steps", 77))
        contracts.append((source.rl_config, contract, yaml_episode_len))

    base_path, base_contract, base_episode_len = contracts[0]
    for path, contract, episode_len in contracts[1:]:
        if contract.compatibility_signature() != base_contract.compatibility_signature():
            raise ValueError(
                f"Incompatible observation/target layout between {base_path} and {path}."
            )
        if requested_episode_len == 0 and episode_len != base_episode_len:
            raise ValueError(
                f"RL configs use different episode lengths: {base_path}={base_episode_len}, "
                f"{path}={episode_len}. Set build.episode_len explicitly only if this is intentional."
            )

    resolved_episode_len = requested_episode_len or base_episode_len
    return base_contract, resolved_episode_len


def make_assignments(
    source_files: list[tuple[SourceConfig, Path, int]],
    *,
    train_envs: set[int] | None,
    val_envs: set[int] | None,
    test_envs: set[int] | None,
    shuffle: bool,
    seed: int,
    train_weight: int,
    val_weight: int,
    test_weight: int,
) -> list[BufferAssignment]:
    explicit = any(value is not None for value in (train_envs, val_envs, test_envs))
    if explicit and any(value is None for value in (train_envs, val_envs, test_envs)):
        raise ValueError("Provide all of --train-envs, --val-envs, and --test-envs, or none.")

    records = list(source_files)
    if explicit:
        split_lookup: dict[int, str] = {}
        assert train_envs is not None and val_envs is not None and test_envs is not None
        if train_envs & val_envs or train_envs & test_envs or val_envs & test_envs:
            raise ValueError("Explicit train/val/test environment ids overlap.")
        split_lookup.update({env_id: "train" for env_id in train_envs})
        split_lookup.update({env_id: "val" for env_id in val_envs})
        split_lookup.update({env_id: "test" for env_id in test_envs})
        selected: list[tuple[SourceConfig, Path, int, str]] = []
        for source, path, env_id in records:
            split = split_lookup.get(env_id)
            if split is not None:
                selected.append((source, path, env_id, split))

        for source in {item[0].source_index: item[0] for item in records}.values():
            available = {env_id for candidate, _path, env_id in records if candidate.source_index == source.source_index}
            missing = set(split_lookup) - available
            if missing:
                raise ValueError(f"Source {source.source_index} is missing explicit env ids: {sorted(missing)}")
    else:
        if shuffle:
            random.Random(seed).shuffle(records)
        split_names = weighted_split_names(
            len(records),
            train_weight=train_weight,
            val_weight=val_weight,
            test_weight=test_weight,
        )
        selected = [
            (source, path, env_id, split)
            for (source, path, env_id), split in zip(records, split_names, strict=True)
        ]

    assignments = [
        BufferAssignment(
            dataset_env_id=index,
            split=split,
            source_index=source.source_index,
            source_env_id=source_env_id,
            source_file=path,
            raw_dir=source.raw_dir,
            buffer_name=source.buffer_name,
            rl_config=source.rl_config,
        )
        for index, (source, path, source_env_id, split) in enumerate(selected)
    ]
    return assignments


def required(value: str | None, env_name: str) -> str:
    resolved = value or os.environ.get(env_name)
    if resolved is None or not resolved.strip():
        raise ValueError(f"Missing --{env_name.lower().replace('_', '-')} or environment variable {env_name}.")
    return resolved.strip()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine one or more RL buffer sources and build a sharded dataset."
    )
    parser.add_argument("--objective")
    parser.add_argument("--folder-name")
    parser.add_argument("--buffer-name", help="One name or comma-separated source names.")
    parser.add_argument("--config", help="One name or comma-separated RL YAML names.")
    parser.add_argument(
        "--raw-dir",
        default="",
        help="Optional one path or comma-separated paths. Defaults from build_defaults.yaml.",
    )
    parser.add_argument(
        "--settings",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "data" / "build_defaults.yaml"),
    )
    parser.add_argument("--job-id", default=os.environ.get("PBS_JOBID", "local"))
    parser.add_argument("--train-envs", default="")
    parser.add_argument("--val-envs", default="")
    parser.add_argument("--test-envs", default="")
    parser.add_argument("--overwrite", action="store_true", help="Override build.overwrite from YAML.")
    return parser


def main(argv: Iterable[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    objective = required(args.objective, "OBJECTIVE")
    folder_name = required(args.folder_name, "FOLDER_NAME")
    buffer_text = required(args.buffer_name, "BUFFER_NAME")
    config_text = required(args.config, "CONFIG")

    settings = load_build_settings(args.settings)
    buffer_names = csv_values(buffer_text)
    config_names = csv_values(config_text)
    raw_dirs = csv_values(args.raw_dir or os.environ.get("RAW_DIR", ""))

    source_count = max(len(buffer_names), len(config_names), len(raw_dirs) or 1)
    buffer_names = broadcast(buffer_names, source_count, "BUFFER_NAME")
    config_names = broadcast(config_names, source_count, "CONFIG")
    if raw_dirs:
        raw_dirs = broadcast(raw_dirs, source_count, "RAW_DIR")
    else:
        source_roots = load_source_roots(Path(args.settings).parent / "source_roots.yaml")
        raw_dirs = []
        for buffer_name in buffer_names:
            resolved = source_roots.resolve(objective, buffer_name) if source_roots else None
            if resolved is None:
                resolved = settings.raw_root_for(objective)
            raw_dirs.append(str(resolved))

    sources: list[SourceConfig] = []
    source_files: list[tuple[SourceConfig, Path, int]] = []
    for source_index, (raw_text, buffer_name, config_name) in enumerate(
        zip(raw_dirs, buffer_names, config_names, strict=True)
    ):
        raw_dir = Path(raw_text).expanduser()
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")
        rl_config = resolve_rl_config(raw_dir, config_name)
        source = SourceConfig(
            source_index=source_index,
            raw_dir=raw_dir,
            buffer_name=buffer_name,
            rl_config=rl_config,
        )
        sources.append(source)
        for path in find_source_files(raw_dir, buffer_name):
            source_files.append((source, path, env_id_from_name(path)))

    contract, resolved_episode_len = validate_contracts(
        sources,
        num_regions=settings.build.num_regions,
        action_classes=settings.build.action_classes,
        include_action_onehot=settings.build.include_action_onehot,
        requested_episode_len=settings.build.episode_len,
    )
    if settings.build.episode_len == 0:
        settings.build.episode_len = resolved_episode_len

    assignments = make_assignments(
        source_files,
        train_envs=parse_env_ids(args.train_envs),
        val_envs=parse_env_ids(args.val_envs),
        test_envs=parse_env_ids(args.test_envs),
        shuffle=settings.split.shuffle,
        seed=settings.split.seed,
        train_weight=settings.split.train_weight,
        val_weight=settings.split.val_weight,
        test_weight=settings.split.test_weight,
    )
    if args.overwrite:
        settings.build.overwrite = True

    request = DatasetBuildRequest(
        objective=objective,
        folder_name=folder_name,
        output_dir=settings.output_root / folder_name.strip().strip(",").replace(" ", ""),
        job_id=str(args.job_id),
        sources=sources,
        split=settings.split,
        build=settings.build,
    )

    counts = {split: sum(item.split == split for item in assignments) for split in ("train", "val", "test")}
    print(f"Objective: {objective}", flush=True)
    print(f"Dataset: {request.folder_name}", flush=True)
    print(f"Sources: {len(sources)}; buffers: {len(assignments)}; split counts: {counts}", flush=True)
    print(
        f"Contract: obs={contract.expected_obs_dim}, "
        f"x={contract.num_regions}x{contract.x_node_width}, "
        f"y={contract.num_regions}x{contract.y_node_width}, "
        f"targets={list(contract.targets)}",
        flush=True,
    )
    return build_dataset(request, assignments, contract=contract)


if __name__ == "__main__":
    main()
