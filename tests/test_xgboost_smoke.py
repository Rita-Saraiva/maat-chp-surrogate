"""End-to-end smoke tests for the XGBoost surrogate on a synthetic dataset.

These reuse the tiny ``npy_memmap_shards`` dataset from the multi-head smoke test
and exercise the full XGBoost path (config -> data layer -> native multi-output
model -> per-head metrics -> results store) for both single- and multi-head runs,
plus a save/load round-trip that must reproduce the fitted predictions exactly.

They are skipped when ``xgboost`` is unavailable or older than 2.0 (native
``multi_output_tree`` requires XGBoost >= 2.0). They run CPU-only and need
``pydantic>=2`` and ``torch``.

Run directly:  python -m tests.test_xgboost_smoke
Or with pytest: pytest tests/test_xgboost_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

# Make the project importable when run as a plain script.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.schema import (  # noqa: E402
    DataTrainingConfig,
    ModelTrainingConfig,
    StandardizationConfig,
    TrainingRunConfig,
    TrainLoopConfig,
    XGBProfileConfig,
)
from tests.test_multihead_smoke import (  # noqa: E402
    COMPONENTS,
    EXPECTED_SLICES,
    EXPECTED_WIDTHS,
    TOTAL_WIDTH,
    X_DIM,
    _build_synthetic_dataset,
)


def _xgboost_available() -> bool:
    try:
        import xgboost
    except ImportError:
        print("SKIP xgboost smoke tests (xgboost not installed)")
        return False
    major = int(str(xgboost.__version__).split(".", 1)[0])
    if major < 2:
        print(f"SKIP xgboost smoke tests (xgboost {xgboost.__version__} < 2.0)")
        return False
    return True


def _model_config() -> ModelTrainingConfig:
    return ModelTrainingConfig(
        family="XGB",
        profile="test_xgb",
        parameters=XGBProfileConfig(
            n_estimators=20,
            max_depth=3,
            learning_rate=0.2,
            subsample=1.0,
            colsample_bytree=1.0,
            tree_method="hist",
            multi_strategy="multi_output_tree",
            early_stopping_rounds=5,
            n_jobs=1,
        ),
    )


def _make_config(
    dataset_dir: Path,
    output_root: Path,
    mode: str,
    loader_mode: str,
    component: str,
    components: list[str] | None,
) -> TrainingRunConfig:
    return TrainingRunConfig(
        job_id="testjob",
        objective="Econ",
        mode=mode,
        project_root=dataset_dir.parent,
        output_root=output_root,
        data=DataTrainingConfig(
            dataset="synthetic",
            dataset_dir=dataset_dir,
            loader_mode=loader_mode,
            component=component,
            components=components,
            batch_size=16,
            num_workers=0,
            pin_memory=False,
        ),
        model=_model_config(),
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


def _val_loader(dataset_dir: Path, loader_mode: str, component: str, components):
    from src.data.training import create_training_dataset, create_training_loader

    dataset = create_training_dataset(
        dataset_dir=dataset_dir,
        split="val",
        loader_mode=loader_mode,
        component=component,
        components=components,
    )
    loader = create_training_loader(
        dataset=dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return dataset, loader


def _run_multi(loader_mode: str) -> None:
    from src.models.xgboost import XGBSurrogate
    from src.train.trainer import train_surrogate

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_dir = _build_synthetic_dataset(
            root / "data", {"train": [40, 24], "val": [24], "test": [24]}
        )
        output_root = root / "out"
        config = _make_config(
            dataset_dir,
            output_root,
            mode="multi",
            loader_mode=loader_mode,
            component="multi_all",
            components=COMPONENTS,
        )
        result = train_surrogate(config)

        validation = result["validation"]
        assert set(validation.keys()) == {"Overall", "Per Head"}
        assert set(validation["Per Head"].keys()) == set(COMPONENTS)
        for name in COMPONENTS:
            head = validation["Per Head"][name]
            assert np.isfinite(head["MSE"]), name
            assert head["Values"] == 24 * EXPECTED_WIDTHS[name], name
        assert validation["Overall"]["Values"] == 24 * TOTAL_WIDTH

        # Per-run record: mode=multi, combined component, heads recorded.
        run_yaml = next((output_root / "results" / "runs").glob("*.yaml"))
        record = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
        assert record["Mode"] == "multi"
        assert record["Component"] == "multi_all"
        assert record["Results"]["Heads"] == EXPECTED_WIDTHS

        # Aggregated store: one row per head.
        import pandas as pd

        table = pd.read_pickle(output_root / "results" / "tables" / "run_history.pkl")
        assert len(table) == len(COMPONENTS)
        assert set(table["Component"]) == set(COMPONENTS)
        assert set(table["Mode"]) == {"multi"}

        # Saved checkpoint round-trips and reproduces the fitted predictions.
        head_output_dims = {name: EXPECTED_WIDTHS[name] for name in COMPONENTS}
        reloaded = XGBSurrogate(
            input_dim=X_DIM,
            output_dim=TOTAL_WIDTH,
            profile=config.model.parameters,
            head_slices=EXPECTED_SLICES,
            head_output_dims=head_output_dims,
        )
        metadata = reloaded.load(result["run_paths"].model_path)
        assert metadata["Heads"] == head_output_dims
        assert reloaded.output_dim == TOTAL_WIDTH
        assert reloaded.head_slices == EXPECTED_SLICES
        # Standardization (config inputs+targets True) round-trips through the checkpoint.
        assert reloaded.standardizer is not None
        assert reloaded.standardizer.standardize_inputs
        assert reloaded.standardizer.standardize_targets

        _, loader = _val_loader(dataset_dir, loader_mode, "multi_all", COMPONENTS)
        reloaded_metrics = reloaded.evaluate(loader)
        np.testing.assert_allclose(
            reloaded_metrics["Overall"]["MSE"],
            validation["Overall"]["MSE"],
            rtol=1e-5,
            atol=1e-6,
        )
    print(f"PASS _run_multi loader_mode={loader_mode}")


def test_multi_xgb_whole() -> None:
    if not _xgboost_available():
        return
    _run_multi("whole")


def test_multi_xgb_chunked() -> None:
    if not _xgboost_available():
        return
    _run_multi("chunked")


def test_single_xgb_whole() -> None:
    if not _xgboost_available():
        return

    from src.models.xgboost import XGBSurrogate
    from src.train.trainer import train_surrogate

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_dir = _build_synthetic_dataset(
            root / "data", {"train": [40], "val": [24], "test": [24]}
        )
        output_root = root / "out"
        config = _make_config(
            dataset_dir,
            output_root,
            mode="single",
            loader_mode="whole",
            component="delay",
            components=None,
        )
        result = train_surrogate(config)

        validation = result["validation"]
        assert "MSE" in validation and "R2" in validation
        assert "Per Head" not in validation
        # delay head width is 3 -> Values = 24 * 3.
        assert validation["Values"] == 24 * EXPECTED_WIDTHS["delay"]

        import pandas as pd

        table = pd.read_pickle(output_root / "results" / "tables" / "run_history.pkl")
        assert len(table) == 1
        assert table.iloc[0]["Component"] == "delay"
        assert table.iloc[0]["Mode"] == "single"

        # Round-trip reproduces the fitted single-head predictions.
        reloaded = XGBSurrogate(
            input_dim=X_DIM,
            output_dim=EXPECTED_WIDTHS["delay"],
            profile=config.model.parameters,
        )
        reloaded.load(result["run_paths"].model_path)
        assert reloaded.standardizer is not None
        _, loader = _val_loader(dataset_dir, "whole", "delay", None)
        reloaded_metrics = reloaded.evaluate(loader)
        np.testing.assert_allclose(
            reloaded_metrics["MSE"], validation["MSE"], rtol=1e-5, atol=1e-6
        )
    print("PASS test_single_xgb_whole")


def main() -> int:
    if not _xgboost_available():
        print("\nXGBOOST SMOKE TESTS SKIPPED")
        return 0
    test_single_xgb_whole()
    test_multi_xgb_whole()
    test_multi_xgb_chunked()
    print("\nALL XGBOOST SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
