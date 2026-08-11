"""Shared helpers for dataset and prediction analysis.

Both ``src.analyze_dataset`` and ``src.plot_predictions`` reconstruct the per-node
feature/target layout from ``manifest.yaml`` (see ``src/data/contract.py``), map
region ids to node indices via ``configs/graph/region_graph.yaml``, and render the
same style of histograms / mean-across-zones scatter plots.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Colour cycle reused for mode overlays and per-zone series.
SERIES_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return cleaned.strip("_") or "unnamed"


def finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def rounded_mode(values: np.ndarray, decimals: int) -> float:
    """Most frequent value after rounding to ``decimals`` places (Option A)."""
    array = finite(values)
    if array.size == 0:
        return math.nan
    rounded = np.round(array, decimals)
    uniques, counts = np.unique(rounded, return_counts=True)
    return float(uniques[int(np.argmax(counts))])


def summarize(name: str, series: str, values: np.ndarray, decimals: int) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = raw[np.isfinite(raw)]
    row: dict[str, Any] = {
        "variable": name,
        "series": series,
        "count": int(raw.size),
        "finite_count": int(valid.size),
        "zero_pct": math.nan,
        "min": math.nan,
        "mode_rounded": math.nan,
        "max": math.nan,
        "mean": math.nan,
        "std": math.nan,
        "median": math.nan,
    }
    if valid.size == 0:
        return row
    row.update(
        {
            "zero_pct": float(np.mean(valid == 0.0) * 100.0),
            "min": float(np.min(valid)),
            "mode_rounded": rounded_mode(valid, decimals),
            "max": float(np.max(valid)),
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
            "median": float(np.median(valid)),
        }
    )
    return row


def reconstruct_x_feature_columns(
    payload: dict[str, Any],
) -> list[tuple[str, int, int, list[str] | None]]:
    """Return ``(name, node_start, node_width, modes)`` for each x feature.

    ``node_start`` / ``node_width`` index the per-node column axis of the stored
    ``x`` array (shape ``[n, num_regions, x_node_width]``). ``modes`` is set only for
    ``zone_water_depth`` (one column per transport mode); ``None`` otherwise.
    """
    contract = payload["contract"]
    num_regions = int(contract["num_regions"])
    modes = [str(mode) for mode in contract.get("modes", [])]

    columns: list[tuple[str, int, int, list[str] | None]] = []
    offset = 0
    for feature in contract["features"]:
        if not feature.get("include_in_x", False):
            continue
        name = str(feature["name"])
        shape = [int(value) for value in feature.get("shape", [])]
        if shape == [1]:
            width = 1
        else:
            length = int(feature["length"])
            if length % num_regions != 0:
                raise ValueError(
                    f"Feature {name} length={length} not divisible by num_regions={num_regions}."
                )
            width = length // num_regions
        feature_modes = modes if (name == "zone_water_depth" and width == len(modes)) else None
        columns.append((name, offset, width, feature_modes))
        offset += width

    if contract.get("include_action_onehot", False):
        action_classes = int(contract["action_classes"])
        columns.append(("action_onehot", offset, action_classes, None))
        offset += action_classes

    x_node_width = int(contract["x_node_width"])
    if offset != x_node_width:
        raise ValueError(
            f"Reconstructed x width {offset} does not match manifest x_node_width={x_node_width}."
        )
    return columns


def target_layout(
    payload: dict[str, Any],
) -> list[tuple[str, list[int], list[str] | None, str]]:
    """Return ``(name, y_columns, modes, type)`` for each target."""
    contract = payload["contract"]
    layout: list[tuple[str, list[int], list[str] | None, str]] = []
    for name, spec in contract["targets"].items():
        y_columns = [int(value) for value in spec["y_columns"]]
        modes = [str(mode) for mode in spec["modes"]] if spec.get("modes") else None
        layout.append((str(name), y_columns, modes, str(spec.get("type", ""))))
    return layout


def zone_indices(
    region_ids: Sequence[int],
    names: dict[int, str],
    requested: Sequence[int],
) -> list[tuple[int, int, str]]:
    """Map requested region ids to ``(zone_id, node_index, label)``.

    Unknown zone ids are reported and skipped so a single bad id never aborts a run.
    """
    id_to_index = {int(region_id): index for index, region_id in enumerate(region_ids)}
    resolved: list[tuple[int, int, str]] = []
    for zone_id in requested:
        zone = int(zone_id)
        if zone not in id_to_index:
            print(f"  WARNING: zone id {zone} not found in region graph; skipping.", flush=True)
            continue
        index = id_to_index[zone]
        resolved.append((zone, index, names.get(zone, str(zone))))
    return resolved


def parse_zone_ids(text: str) -> list[int]:
    if not text or not text.strip():
        return []
    ids: list[int] = []
    for token in re.split(r"[,\s]+", text.strip()):
        if not token:
            continue
        ids.append(int(token))
    return ids


def save_histogram(values: np.ndarray, title: str, path: Path, bins: int) -> Path | None:
    array = finite(values)
    if array.size == 0:
        print(f"  No finite values for {title}; skipping histogram.", flush=True)
        return None
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.hist(array, bins=bins)
    axis.set_title(title)
    axis.set_xlabel("value")
    axis.set_ylabel("count")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def _subsample(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]
    if max_points > 0 and x_valid.size > max_points:
        selection = rng.choice(x_valid.size, size=max_points, replace=False)
        x_valid = x_valid[selection]
        y_valid = y_valid[selection]
    return x_valid, y_valid


def save_scatter(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
    path: Path,
    max_points: int,
    rng: np.random.Generator,
    identity_line: bool = False,
) -> Path | None:
    """One scatter figure overlaying ``(label, x, y)`` series in distinct colours."""
    figure, axis = plt.subplots(figsize=(9, 6))
    plotted = 0
    all_values: list[np.ndarray] = []
    for index, (label, x_values, y_values) in enumerate(series):
        x_flat = np.asarray(x_values, dtype=np.float64).reshape(-1)
        y_flat = np.asarray(y_values, dtype=np.float64).reshape(-1)
        if x_flat.size != y_flat.size:
            raise ValueError(f"Series {label!r} x/y size mismatch: {x_flat.size} vs {y_flat.size}")
        x_sample, y_sample = _subsample(x_flat, y_flat, max_points, rng)
        if x_sample.size == 0:
            continue
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        axis.scatter(x_sample, y_sample, s=6, alpha=0.3, linewidths=0, color=color, label=label)
        all_values.append(x_sample)
        all_values.append(y_sample)
        plotted += 1

    if plotted == 0:
        plt.close(figure)
        print(f"  No finite points for {title}; skipping scatter.", flush=True)
        return None

    if identity_line and all_values:
        pooled = np.concatenate(all_values)
        low = float(np.min(pooled))
        high = float(np.max(pooled))
        axis.plot([low, high], [low, high], color="black", linewidth=1.0, linestyle="--", label="y = x")

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.2)
    if len(series) > 1 or identity_line:
        axis.legend(fontsize=8, markerscale=2)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    import csv

    fieldnames = [
        "variable",
        "series",
        "count",
        "finite_count",
        "zero_pct",
        "min",
        "mode_rounded",
        "max",
        "mean",
        "std",
        "median",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6g}"


def write_summary_text(rows: list[dict[str, Any]], path: Path, header_lines: list[str]) -> None:
    lines = list(header_lines)
    lines.append("")
    lines.append(
        f"{'variable':<22} {'series':<12} {'min':>12} {'mode':>12} {'max':>12} "
        f"{'mean':>12} {'std':>12} {'median':>12} {'%zero':>8}"
    )
    lines.append("-" * 118)
    for row in rows:
        lines.append(
            f"{str(row['variable']):<22} {str(row['series']):<12} "
            f"{_format(row['min']):>12} {_format(row['mode_rounded']):>12} "
            f"{_format(row['max']):>12} {_format(row['mean']):>12} "
            f"{_format(row['std']):>12} {_format(row['median']):>12} "
            f"{_format(row['zero_pct']):>7}%"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
