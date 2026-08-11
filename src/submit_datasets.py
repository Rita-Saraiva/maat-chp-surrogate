#!/usr/bin/env python3
"""Submit one or more configured dataset builds to PBS.

Examples:
    python src/submit_datasets.py econ_rcp_26
    python src/submit_datasets.py econ_combined qol_jumbo
    python src/submit_datasets.py --all
    python src/submit_datasets.py --all --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml


def load_catalog(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError(f"Invalid dataset catalog: {path}")
    return payload


def build_environment(spec: dict[str, Any]) -> dict[str, str]:
    sources = spec.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Dataset {spec.get('folder_name')} has no sources")

    buffers: list[str] = []
    configs: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each source must be a mapping")
        buffer_name = str(source.get("buffer_name", "")).strip()
        config = str(source.get("config", "")).strip()
        if not buffer_name or not config:
            raise ValueError(f"Incomplete source: {source}")
        buffers.append(buffer_name)
        configs.append(config)

    env = os.environ.copy()
    env.update(
        {
            "OBJECTIVE": str(spec["objective"]),
            "FOLDER_NAME": str(spec["folder_name"]),
            "BUFFER_NAME": ",".join(buffers),
            "CONFIG": ",".join(configs),
        }
    )
    return env


def qsub_command(pbs_script: str) -> list[str]:
    return [
        "qsub",
        "-v",
        "OBJECTIVE,FOLDER_NAME,BUFFER_NAME,CONFIG",
        pbs_script,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit configured dataset builds.")
    parser.add_argument("datasets", nargs="*", help="Dataset names from dataset_builds.yaml")
    parser.add_argument("--all", action="store_true", help="Submit every configured dataset")
    parser.add_argument("--dry-run", action="store_true", help="Print without submitting")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/data/dataset_builds.yaml"),
    )
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    definitions: dict[str, Any] = catalog["datasets"]
    pbs_script = str(catalog.get("pbs_script", "jobs/build_dataset.pbs"))

    if args.all:
        selected = list(definitions)
    elif args.datasets:
        selected = args.datasets
    else:
        parser.error("Provide dataset names or --all")

    unknown = [name for name in selected if name not in definitions]
    if unknown:
        raise KeyError(f"Unknown datasets: {unknown}. Available: {list(definitions)}")

    for name in selected:
        spec = definitions[name]
        env = build_environment(spec)
        command = qsub_command(pbs_script)
        print(
            f"{name}: OBJECTIVE={env['OBJECTIVE']} FOLDER_NAME={env['FOLDER_NAME']} "
            f"BUFFER_NAME={env['BUFFER_NAME']} CONFIG={env['CONFIG']}",
            flush=True,
        )
        if args.dry_run:
            print("  " + " ".join(command), flush=True)
            continue
        result = subprocess.run(command, env=env, check=True, text=True, capture_output=True)
        print(f"  job_id={result.stdout.strip()}", flush=True)


if __name__ == "__main__":
    main()
