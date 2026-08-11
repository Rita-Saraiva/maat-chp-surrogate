#!/usr/bin/env python3
"""Plot datasize-robustness curves (training fraction vs test performance).

Reads the ``run_fraction`` table produced by fraction sweeps (see
:func:`src.results.training_results.append_fraction_history`) and renders, for
each test metric (R2, RMSE, MAE):

* one figure per model + parameter profile (a single curve), and
* one figure per model overlaying every parameter profile (one curve each).

Each figure devotes a separate subplot to each metric. PNGs are written under
``results/analysis/robustness``.

Examples::

    python -m src.experiments.plot_robustness
    python -m src.experiments.plot_robustness --dataset qol_combined --model MLP
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# (Test-metric key, axis label, higher-is-better)
METRICS: list[tuple[str, str, bool]] = [
    ("R2", "Test R\u00b2", True),
    ("RMSE", "Test RMSE", False),
    ("MAE", "Test MAE", False),
]

# Columns identifying a comparable run, excluding the swept fraction and profile.
GROUP_COLUMNS = ["Dataset", "Objective", "Mode", "Component", "Model"]


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("_") or "na"


def load_fraction_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(
            f"Fraction table does not exist: {table_path}. Run a TRAIN_FRACTION sweep first."
        )
    frame = pd.read_pickle(table_path)
    if frame.empty:
        raise ValueError(f"Fraction table is empty: {table_path}")
    return frame


def _profile(row: pd.Series) -> str:
    parameters = row.get("Parameters")
    if isinstance(parameters, dict):
        return str(parameters.get("Profile", "unknown"))
    return "unknown"


def _test_metric(row: pd.Series, key: str) -> float | None:
    test = row.get("Test")
    if not isinstance(test, dict) or key not in test:
        return None
    try:
        return float(test[key])
    except (TypeError, ValueError):
        return None


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["Profile"] = prepared.apply(_profile, axis=1)
    prepared["Fraction"] = pd.to_numeric(prepared["Fraction"], errors="coerce")
    for key, _, _ in METRICS:
        prepared[key] = prepared.apply(lambda row, k=key: _test_metric(row, k), axis=1)
    return prepared.dropna(subset=["Fraction"])


def _apply_filters(frame: pd.DataFrame, filters: dict[str, str | None]) -> pd.DataFrame:
    filtered = frame
    for column, value in filters.items():
        if value is not None:
            filtered = filtered[filtered[column] == value]
    return filtered


def _new_figure(title: str) -> tuple[Any, list[Any]]:
    figure, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 5))
    if len(METRICS) == 1:
        axes = [axes]
    figure.suptitle(title)
    return figure, list(axes)


def _finish_axis(axis: Any, label: str, higher_better: bool) -> None:
    trend = "higher is better" if higher_better else "lower is better"
    axis.set_xlabel("Training fraction")
    axis.set_ylabel(label)
    axis.set_title(f"{label} ({trend})")
    axis.grid(True, alpha=0.3)


def plot_per_parameter(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """One figure per (group + profile): a single curve across fractions."""
    written: list[Path] = []
    keys = GROUP_COLUMNS + ["Profile"]
    for values, group in frame.groupby(keys):
        group = group.sort_values("Fraction")
        label_map = dict(zip(keys, values))
        title = (
            f"{label_map['Model']} / {label_map['Profile']} — "
            f"{label_map['Dataset']} / {label_map['Objective']} / "
            f"{label_map['Component']} ({label_map['Mode']})"
        )
        figure, axes = _new_figure(title)
        for axis, (key, label, higher_better) in zip(axes, METRICS):
            axis.plot(group["Fraction"], group[key], marker="o")
            _finish_axis(axis, label, higher_better)
        figure.tight_layout()
        name = "_".join(_safe(label_map[key]) for key in keys)
        out_path = out_dir / f"per_parameter_{name}.png"
        figure.savefig(out_path, dpi=150)
        plt.close(figure)
        written.append(out_path)
    return written


def plot_per_model(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    """One figure per model group: one curve per parameter profile overlaid."""
    written: list[Path] = []
    for values, group in frame.groupby(GROUP_COLUMNS):
        label_map = dict(zip(GROUP_COLUMNS, values))
        title = (
            f"{label_map['Model']} — all parameters — "
            f"{label_map['Dataset']} / {label_map['Objective']} / "
            f"{label_map['Component']} ({label_map['Mode']})"
        )
        figure, axes = _new_figure(title)
        for axis, (key, label, higher_better) in zip(axes, METRICS):
            for profile, profile_group in group.groupby("Profile"):
                profile_group = profile_group.sort_values("Fraction")
                axis.plot(
                    profile_group["Fraction"],
                    profile_group[key],
                    marker="o",
                    label=str(profile),
                )
            _finish_axis(axis, label, higher_better)
            axis.legend(title="Parameters", fontsize="small")
        figure.tight_layout()
        name = "_".join(_safe(label_map[key]) for key in GROUP_COLUMNS)
        out_path = out_dir / f"per_model_{name}.png"
        figure.savefig(out_path, dpi=150)
        plt.close(figure)
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fraction-vs-robustness curves.")
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("results") / "tables" / "run_fraction.pkl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "analysis" / "robustness",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--component", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    frame = _prepare(load_fraction_table(args.table))
    frame = _apply_filters(
        frame,
        {
            "Dataset": args.dataset,
            "Objective": args.objective,
            "Mode": args.mode,
            "Component": args.component,
            "Model": args.model,
        },
    )
    if frame.empty:
        raise ValueError("No rows match the requested filters")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = plot_per_parameter(frame, args.out_dir) + plot_per_model(frame, args.out_dir)

    print(f"Wrote {len(written)} figure(s) to {args.out_dir}:", flush=True)
    for path in written:
        print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
