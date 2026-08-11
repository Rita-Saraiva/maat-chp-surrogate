"""Analyze processed surrogate datasets (stats, histograms, water-depth plots).

Reads the ``npy_memmap_shards`` dataset produced by the dataset builder directly
from disk (``manifest.yaml`` + shard ``.npy`` files) and, for each requested split,
writes a summary statistics table, per-feature / per-target histograms, and point
plots of every target against ``zone_water_depth`` (mean across zones, one point per
time step). Everything is driven by environment variables so it can run unchanged in
the container via ``jobs/analyze_dataset.pbs``.

Environment variables:
    DATASET        Dataset directory name under ``<DATA_ROOT>`` (required).
    SPLITS         Comma-separated splits to analyze (default ``train,val,test``).
    ZONES          Optional region ids for per-zone series plots (default: none).
    MAX_SAMPLES    Cap on samples per split (0 / unset = all).
    MAX_SHARDS     Cap on shards read per split (0 / unset = all).
    HIST_BINS      Histogram bin count (default 80).
    MAX_SCATTER_POINTS  Cap points per scatter (default 200000; 0 = all).
    MODE_DECIMALS  Rounding places for the "mode" statistic (default 3).
    SEED           Random seed for scatter subsampling (default 42).
    PROJECT_ROOT / DATA_ROOT / OUTPUT_ROOT  Path roots (container-friendly defaults).

Run: ``python -u -m src.analyze_dataset``
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from src.data.dataset import DatasetManifest
from src.data.graph import load_region_graph
from src.eval import analysis_common as ac


def _int_env(name: str, default: int) -> int:
    text = os.environ.get(name, "").strip()
    if not text:
        return default
    value = int(text)
    return value


def _load_split_arrays(
    manifest: DatasetManifest,
    split: str,
    max_samples: int,
    max_shards: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate shard ``x``/``y`` arrays for ``split`` into ``[n, R, W]`` blocks."""
    shards = manifest.shards(split)
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    total = 0
    for index, entry in enumerate(shards):
        if max_shards > 0 and index >= max_shards:
            break
        x_array = np.load(manifest.dataset_dir / entry["x"])
        y_array = np.load(manifest.dataset_dir / entry["y"])
        x_blocks.append(np.asarray(x_array, dtype=np.float64))
        y_blocks.append(np.asarray(y_array, dtype=np.float64))
        total += int(x_array.shape[0])
        if max_samples > 0 and total >= max_samples:
            break
    if not x_blocks:
        raise ValueError(f"No shards found for split={split!r} in {manifest.dataset_dir}.")
    x_full = np.concatenate(x_blocks, axis=0)
    y_full = np.concatenate(y_blocks, axis=0)
    if max_samples > 0 and x_full.shape[0] > max_samples:
        x_full = x_full[:max_samples]
        y_full = y_full[:max_samples]
    return x_full, y_full


def _feature_series(
    x_full: np.ndarray,
    name: str,
    start: int,
    width: int,
    modes: list[str] | None,
) -> list[tuple[str, np.ndarray]]:
    """Return ``(series_label, values[n*R])`` for one x feature.

    Mode features expose one series per mode; everything else is pooled into a
    single series (multi-column features like ``zone_modifiers`` stay compact).
    """
    block = x_full[:, :, start : start + width]
    if modes is not None:
        return [(f"{name}[{mode}]", block[:, :, col].reshape(-1)) for col, mode in enumerate(modes)]
    return [(name, block.reshape(-1))]


def _analyze_split(
    manifest: DatasetManifest,
    split: str,
    output_dir: Path,
    zones: list[tuple[int, int, str]],
    hist_bins: int,
    max_scatter_points: int,
    mode_decimals: int,
    max_samples: int,
    max_shards: int,
    rng: np.random.Generator,
) -> None:
    payload = manifest.payload
    print(f"\n=== Split: {split} ===", flush=True)
    x_full, y_full = _load_split_arrays(manifest, split, max_samples, max_shards)
    num_samples, num_regions, _ = x_full.shape
    print(f"Loaded x={x_full.shape} y={y_full.shape}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    x_features = ac.reconstruct_x_feature_columns(payload)
    targets = ac.target_layout(payload)
    modes = [str(mode) for mode in payload["contract"].get("modes", [])]

    water = next((item for item in x_features if item[0] == "zone_water_depth"), None)
    if water is None:
        raise KeyError("Dataset has no zone_water_depth feature in x.")
    _, water_start, water_width, water_modes = water
    water_block = x_full[:, :, water_start : water_start + water_width]  # [n, R, num_modes]
    water_modes = water_modes or [f"mode_{i}" for i in range(water_width)]

    summary_rows: list[dict[str, Any]] = []

    # Input feature statistics + histograms.
    for name, start, width, feature_modes in x_features:
        for label, values in _feature_series(x_full, name, start, width, feature_modes):
            summary_rows.append(ac.summarize(name, label, values, mode_decimals))
            ac.save_histogram(
                values,
                title=f"{split}: {label} (input)",
                path=output_dir / f"hist_input_{ac.safe_name(label)}.png",
                bins=hist_bins,
            )

    # Target statistics + histograms + water-depth scatter plots.
    for name, y_columns, target_modes, target_type in targets:
        target_block = y_full[:, :, y_columns]  # [n, R, len(cols)]
        if target_modes is not None:
            series_defs = [
                (f"{name}[{mode}]", target_block[:, :, col], col)
                for col, mode in enumerate(target_modes)
            ]
        else:
            series_defs = [(name, target_block[:, :, 0], None)]

        for label, column_block, mode_col in series_defs:
            summary_rows.append(ac.summarize(name, label, column_block.reshape(-1), mode_decimals))
            ac.save_histogram(
                column_block[column_block != 0.0],
                title=f"{split}: {label} (target, nonzero)",
                path=output_dir / f"hist_target_{ac.safe_name(label)}.png",
                bins=hist_bins,
            )

        _plot_target_vs_water(
            split=split,
            name=name,
            target_block=target_block,
            target_modes=target_modes,
            water_block=water_block,
            water_modes=water_modes,
            zones=zones,
            output_dir=output_dir,
            max_scatter_points=max_scatter_points,
            rng=rng,
        )

    header = [
        f"Dataset analysis — {payload['dataset']['name']} ({split})",
        f"manifest: {manifest.path}",
        f"samples: {num_samples}   regions: {num_regions}",
        f"modes: {modes}",
        "mode_rounded = most frequent value after rounding "
        f"to {mode_decimals} decimals",
    ]
    ac.write_summary_csv(summary_rows, output_dir / "summary_statistics.csv")
    ac.write_summary_text(summary_rows, output_dir / "summary_statistics.txt", header)
    print(f"Wrote summary + plots to {output_dir}", flush=True)


def _plot_target_vs_water(
    split: str,
    name: str,
    target_block: np.ndarray,
    target_modes: list[str] | None,
    water_block: np.ndarray,
    water_modes: list[str],
    zones: list[tuple[int, int, str]],
    output_dir: Path,
    max_scatter_points: int,
    rng: np.random.Generator,
) -> None:
    """Point plots of a target vs zone_water_depth (mean across zones per sample)."""
    if target_modes is not None:
        # Per-mode plots: match each target mode to the same water-depth mode.
        for col, mode in enumerate(target_modes):
            water_index = water_modes.index(mode) if mode in water_modes else min(col, water_block.shape[2] - 1)
            x_mean = water_block[:, :, water_index].mean(axis=1)
            y_mean = target_block[:, :, col].mean(axis=1)
            ac.save_scatter(
                [(f"{mode} (mean over zones)", x_mean, y_mean)],
                xlabel=f"zone_water_depth [{mode}]",
                ylabel=name,
                title=f"{split}: {name}[{mode}] vs water depth (mean over zones)",
                path=output_dir / f"point_{ac.safe_name(name)}_{ac.safe_name(mode)}_meanzones.png",
                max_points=max_scatter_points,
                rng=rng,
            )
        # Aggregate plot: all modes overlaid in distinct colours.
        aggregate = []
        for col, mode in enumerate(target_modes):
            water_index = water_modes.index(mode) if mode in water_modes else min(col, water_block.shape[2] - 1)
            aggregate.append(
                (mode, water_block[:, :, water_index].mean(axis=1), target_block[:, :, col].mean(axis=1))
            )
        ac.save_scatter(
            aggregate,
            xlabel="zone_water_depth (per mode)",
            ylabel=name,
            title=f"{split}: {name} vs water depth (mean over zones, all modes)",
            path=output_dir / f"point_{ac.safe_name(name)}_aggregate_meanzones.png",
            max_points=max_scatter_points,
            rng=rng,
        )
    else:
        # Region-only target: plot against the mean water depth over all modes.
        x_mean = water_block.mean(axis=(1, 2))
        y_mean = target_block[:, :, 0].mean(axis=1)
        ac.save_scatter(
            [("mean over zones", x_mean, y_mean)],
            xlabel="zone_water_depth (mean over modes)",
            ylabel=name,
            title=f"{split}: {name} vs water depth (mean over zones)",
            path=output_dir / f"point_{ac.safe_name(name)}_meanzones.png",
            max_points=max_scatter_points,
            rng=rng,
        )

    if not zones:
        return

    # Per-zone series: one coloured series per selected zone (no averaging).
    if target_modes is not None:
        for col, mode in enumerate(target_modes):
            water_index = water_modes.index(mode) if mode in water_modes else min(col, water_block.shape[2] - 1)
            series = [
                (f"{label} ({zone_id})", water_block[:, node, water_index], target_block[:, node, col])
                for zone_id, node, label in zones
            ]
            ac.save_scatter(
                series,
                xlabel=f"zone_water_depth [{mode}]",
                ylabel=name,
                title=f"{split}: {name}[{mode}] vs water depth (per zone)",
                path=output_dir / f"point_{ac.safe_name(name)}_{ac.safe_name(mode)}_zones.png",
                max_points=max_scatter_points,
                rng=rng,
            )
    else:
        series = [
            (f"{label} ({zone_id})", water_block[:, node, :].mean(axis=1), target_block[:, node, 0])
            for zone_id, node, label in zones
        ]
        ac.save_scatter(
            series,
            xlabel="zone_water_depth (mean over modes)",
            ylabel=name,
            title=f"{split}: {name} vs water depth (per zone)",
            path=output_dir / f"point_{ac.safe_name(name)}_zones.png",
            max_points=max_scatter_points,
            rng=rng,
        )


def main() -> None:
    env = os.environ
    dataset = env.get("DATASET", "").strip()
    if not dataset:
        raise ValueError("DATASET is required (dataset directory name under DATA_ROOT).")

    project_root = Path(env.get("PROJECT_ROOT", "/mnt/project")).expanduser()
    data_root = Path(env.get("DATA_ROOT", str(project_root / "data"))).expanduser()
    output_root = Path(env.get("OUTPUT_ROOT", str(project_root))).expanduser()
    dataset_dir = data_root / dataset

    splits = [item.strip() for item in env.get("SPLITS", "train,val,test").split(",") if item.strip()]
    zone_ids = ac.parse_zone_ids(env.get("ZONES", ""))
    hist_bins = max(1, _int_env("HIST_BINS", 80))
    max_scatter_points = max(0, _int_env("MAX_SCATTER_POINTS", 200_000))
    mode_decimals = _int_env("MODE_DECIMALS", 3)
    max_samples = max(0, _int_env("MAX_SAMPLES", 0))
    max_shards = max(0, _int_env("MAX_SHARDS", 0))
    seed = _int_env("SEED", 42)

    manifest = DatasetManifest(dataset_dir)

    zones: list[tuple[int, int, str]] = []
    if zone_ids:
        graph = load_region_graph(search_roots=[project_root, dataset_dir])
        zones = ac.zone_indices(graph.region_ids, graph.names, zone_ids)
        print(f"Selected zones: {[f'{label} ({zid})' for zid, _, label in zones]}", flush=True)

    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"Manifest: {manifest.path}", flush=True)
    print(f"Splits: {splits}", flush=True)

    for split in splits:
        output_dir = output_root / "results" / "analysis" / "dataset" / dataset / split
        rng = np.random.default_rng(seed)
        _analyze_split(
            manifest=manifest,
            split=split,
            output_dir=output_dir,
            zones=zones,
            hist_bins=hist_bins,
            max_scatter_points=max_scatter_points,
            mode_decimals=mode_decimals,
            max_samples=max_samples,
            max_shards=max_shards,
            rng=rng,
        )

    print("\nDataset analysis complete.", flush=True)


if __name__ == "__main__":
    main()
