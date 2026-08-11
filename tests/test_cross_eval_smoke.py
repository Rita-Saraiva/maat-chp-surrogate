"""End-to-end smoke tests for cross-dataset evaluation.

Trains a tiny MLP on one synthetic dataset (single- and multi-head), then reloads
the resulting checkpoint and scores it on the test split of a *second* synthetic
dataset with the same input/output schema via ``evaluate_cross_dataset``. Asserts
the ``cross_eval.pkl`` store is written with the expected schema and metrics.

Run directly:  python -m tests.test_cross_eval_smoke
Or with pytest: pytest tests/test_cross_eval_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project importable when run as a plain script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.test_multihead_smoke import (  # noqa: E402
    COMPONENTS,
    _build_synthetic_dataset,
    _make_config,
)

from src.eval.cross_dataset import EvaluationRunConfig, evaluate_cross_dataset  # noqa: E402
from src.results.training_results import CROSS_EVAL_COLUMNS  # noqa: E402
from src.train.trainer import train_surrogate  # noqa: E402


def _eval_config(
    output_root: Path,
    project_root: Path,
    eval_dataset_dir: Path,
    mode: str,
    component: str,
    components: list[str] | None,
) -> EvaluationRunConfig:
    return EvaluationRunConfig(
        job_id="evaljob",
        objective="Econ",
        mode=mode,
        family="MLP",
        profile="test_mlp",
        component=component,
        components=components,
        source_job_id="testjob",
        train_dataset="synthetic",
        eval_dataset="synthetic_eval",
        eval_dataset_dir=eval_dataset_dir,
        project_root=project_root,
        output_root=output_root,
        loader_mode="chunked",
        batch_size=16,
        num_workers=0,
        pin_memory=False,
    )


def test_cross_eval_single() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_dir = _build_synthetic_dataset(
            root / "data" / "synthetic",
            {"train": [40, 24], "val": [24], "test": [24]},
            seed=0,
        )
        eval_dir = _build_synthetic_dataset(
            root / "data" / "synthetic_eval",
            {"train": [16], "val": [16], "test": [20]},
            seed=7,
        )
        output_root = root / "out"

        train_config = _make_config(
            train_dir,
            output_root,
            family="MLP",
            mode="single",
            loader_mode="chunked",
            component="delay",
            components=None,
        )
        train_surrogate(train_config)

        result = evaluate_cross_dataset(
            _eval_config(
                output_root,
                project_root=root,
                eval_dataset_dir=eval_dir,
                mode="single",
                component="delay",
                components=None,
            )
        )
        assert np.isfinite(result["test"]["R2"])

        table = pd.read_pickle(output_root / "results" / "tables" / "cross_eval.pkl")
        assert list(table.columns) == CROSS_EVAL_COLUMNS
        assert len(table) == 1
        row = table.iloc[0]
        assert row["Dataset"] == "synthetic_eval"
        assert row["Train Dataset"] == "synthetic"
        assert row["Source Job Id"] == "testjob"
        assert row["Component"] == "delay"
        assert row["Mode"] == "single"
        for metric in ("R2", "MSE", "MAE"):
            assert metric in row["Test"]

        # cross_eval store is separate from the training run-history store.
        assert (output_root / "results" / "tables" / "run_history.pkl").is_file()
    print("PASS test_cross_eval_single")


def test_cross_eval_multi() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_dir = _build_synthetic_dataset(
            root / "data" / "synthetic",
            {"train": [40, 24], "val": [24], "test": [24]},
            seed=0,
        )
        eval_dir = _build_synthetic_dataset(
            root / "data" / "synthetic_eval",
            {"train": [16], "val": [16], "test": [20]},
            seed=7,
        )
        output_root = root / "out"

        train_config = _make_config(
            train_dir,
            output_root,
            family="MLP",
            mode="multi",
            loader_mode="chunked",
            component="multi_all",
            components=COMPONENTS,
        )
        train_surrogate(train_config)

        result = evaluate_cross_dataset(
            _eval_config(
                output_root,
                project_root=root,
                eval_dataset_dir=eval_dir,
                mode="multi",
                component="multi_all",
                components=COMPONENTS,
            )
        )
        assert set(result["test"].keys()) == {"Overall", "Per Head"}

        table = pd.read_pickle(output_root / "results" / "tables" / "cross_eval.pkl")
        assert list(table.columns) == CROSS_EVAL_COLUMNS
        assert len(table) == len(COMPONENTS)
        assert set(table["Component"]) == set(COMPONENTS)
        assert set(table["Mode"]) == {"multi"}
        assert set(table["Train Dataset"]) == {"synthetic"}
        assert set(table["Source Job Id"]) == {"testjob"}
        for _, row in table.iterrows():
            for metric in ("R2", "MSE", "MAE"):
                assert metric in row["Test"]
    print("PASS test_cross_eval_multi")


def test_cross_eval_schema_mismatch_raises() -> None:
    """A dimension mismatch between checkpoint and eval dataset fails fast."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train_dir = _build_synthetic_dataset(
            root / "data" / "synthetic",
            {"train": [40], "val": [24], "test": [24]},
            seed=0,
        )
        eval_dir = _build_synthetic_dataset(
            root / "data" / "synthetic_eval",
            {"train": [16], "val": [16], "test": [20]},
            seed=7,
        )
        output_root = root / "out"

        train_config = _make_config(
            train_dir,
            output_root,
            family="MLP",
            mode="single",
            loader_mode="chunked",
            component="delay",
            components=None,
        )
        train_surrogate(train_config)

        # A single-head checkpoint (output_dim = one component) cannot be scored as
        # if it were multi-head (output_dim = all components concatenated).
        raised = False
        try:
            evaluate_cross_dataset(
                _eval_config(
                    output_root,
                    project_root=root,
                    eval_dataset_dir=eval_dir,
                    mode="single",
                    component="delay",
                    components=COMPONENTS,  # forces a wider output layout
                )
            )
        except ValueError as exc:
            raised = "does not match" in str(exc)
        assert raised, "expected a schema-mismatch ValueError"
    print("PASS test_cross_eval_schema_mismatch_raises")


if __name__ == "__main__":
    test_cross_eval_single()
    test_cross_eval_multi()
    test_cross_eval_schema_mismatch_raises()
    print("ALL cross-eval smoke tests passed")
