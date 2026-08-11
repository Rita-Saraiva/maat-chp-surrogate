"""Smoke test for datasize-robustness (train_fraction) training.

Fabricates a tiny multi-shard ``npy_memmap_shards`` dataset and verifies that a
``train_fraction`` run trains on only the first k whole shards of the TRAIN split
while validation/test stay full, and that the dedicated ``run_fraction`` table is
written with the fraction as its own column.

Run directly:  python -m tests.test_fraction_smoke
Or with pytest: pytest tests/test_fraction_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.schema import (  # noqa: E402
    DataTrainingConfig,
    MLPProfileConfig,
    ModelTrainingConfig,
    StandardizationConfig,
    TrainingRunConfig,
    TrainLoopConfig,
)
from tests.test_multihead_smoke import _build_synthetic_dataset  # noqa: E402

# 4 equal train shards (80 rows) makes "first k shards" arithmetic exact.
TRAIN_SHARDS = [20, 20, 20, 20]
VAL_ROWS = 24
TEST_ROWS = 24


def _make_config(
    dataset_dir: Path,
    output_root: Path,
    train_fraction: float | None,
) -> TrainingRunConfig:
    return TrainingRunConfig(
        job_id="fracjob",
        objective="Econ",
        mode="single",
        project_root=dataset_dir.parent,
        output_root=output_root,
        data=DataTrainingConfig(
            dataset="synthetic",
            dataset_dir=dataset_dir,
            loader_mode="chunked",
            component="delay",
            batch_size=16,
            num_workers=0,
            pin_memory=False,
            train_fraction=train_fraction,
        ),
        model=ModelTrainingConfig(
            family="MLP",
            profile="test_mlp",
            parameters=MLPProfileConfig(hidden_dims=[16, 8], dropout=0.0),
        ),
        training=TrainLoopConfig(
            epochs=2,
            patience=2,
            print_every=1,
            seed=0,
            device="cpu",
            deterministic=True,
            standardization=StandardizationConfig(inputs=True, targets=True),
        ),
    )


def test_train_fraction_uses_first_k_shards() -> None:
    from src.train.trainer import train_surrogate

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_dir = _build_synthetic_dataset(
            root / "data",
            {"train": TRAIN_SHARDS, "val": [VAL_ROWS], "test": [TEST_ROWS]},
        )
        output_root = root / "out"
        # fraction 0.5 of 4 shards -> first 2 shards -> 40 rows.
        config = _make_config(dataset_dir, output_root, train_fraction=0.5)
        result = train_surrogate(config)

        run_yaml = next((output_root / "results" / "runs").glob("*.yaml"))
        record = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
        data_stats = record["Results"]["Data"]
        assert data_stats["Train Rows"] == 40, data_stats
        assert data_stats["Validation Rows"] == VAL_ROWS, data_stats
        assert data_stats["Test Rows"] == TEST_ROWS, data_stats
        assert record["Parameters"]["Data Loader"]["Train Fraction"] == 0.5

        # Dedicated fraction table exists with Fraction as a column.
        fraction_table = pd.read_pickle(
            output_root / "results" / "tables" / "run_fraction.pkl"
        )
        assert "Fraction" in fraction_table.columns
        assert len(fraction_table) == 1
        row = fraction_table.iloc[0]
        assert row["Fraction"] == 0.5
        assert row["Component"] == "delay"
        assert row["Mode"] == "single"

        # Policy metrics and the split/timing columns are present.
        assert {"Validation", "Test", "Training Time (s)"}.issubset(fraction_table.columns)
        assert "Policy Kendall Tau" in row["Test"]
        assert "Policy Aggregate Kendall Tau" in row["Test"]

        # The main run-history table is unaffected in schema (no Fraction column).
        history_table = pd.read_pickle(
            output_root / "results" / "tables" / "run_history.pkl"
        )
        assert "Fraction" not in history_table.columns
    print("PASS test_train_fraction_uses_first_k_shards")


def test_no_fraction_skips_fraction_table() -> None:
    from src.train.trainer import train_surrogate

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_dir = _build_synthetic_dataset(
            root / "data",
            {"train": TRAIN_SHARDS, "val": [VAL_ROWS], "test": [TEST_ROWS]},
        )
        output_root = root / "out"
        config = _make_config(dataset_dir, output_root, train_fraction=None)
        result = train_surrogate(config)
        assert result is not None

        run_yaml = next((output_root / "results" / "runs").glob("*.yaml"))
        record = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
        # Full training set (80 rows) when no fraction is requested.
        assert record["Results"]["Data"]["Train Rows"] == sum(TRAIN_SHARDS)
        assert record["Parameters"]["Data Loader"]["Train Fraction"] is None

        assert not (output_root / "results" / "tables" / "run_fraction.pkl").exists()
    print("PASS test_no_fraction_skips_fraction_table")


if __name__ == "__main__":
    test_train_fraction_uses_first_k_shards()
    test_no_fraction_skips_fraction_table()
    print("All fraction smoke tests passed.")
