"""End-to-end smoke tests for multi-head training on a synthetic dataset.

These tests fabricate a tiny ``npy_memmap_shards`` dataset in a temp directory and
exercise the full multi-head path (config -> data layer -> shared-trunk model ->
per-head metrics -> results store) for MLP/Auto/VAE, plus the unchanged
single-head path. They run CPU-only and need ``pydantic>=2`` and ``torch``.

Run directly:  python -m tests.test_multihead_smoke
Or with pytest: pytest tests/test_multihead_smoke.py
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
    AutoProfileConfig,
    DataTrainingConfig,
    GPProfileConfig,
    MLPProfileConfig,
    ModelTrainingConfig,
    StandardizationConfig,
    TrainingRunConfig,
    TrainLoopConfig,
    VAEProfileConfig,
)

NUM_REGIONS = 3
Y_NODE_WIDTH = 4
X_DIM = 5
COMPONENTS = ["delay", "cancel", "infra"]
# delay -> col [0] (width 3), cancel -> col [1] (width 3), infra -> cols [2,3] (width 6)
EXPECTED_SLICES = {"delay": (0, 3), "cancel": (3, 6), "infra": (6, 12)}
EXPECTED_WIDTHS = {"delay": 3, "cancel": 3, "infra": 6}
TOTAL_WIDTH = sum(EXPECTED_WIDTHS.values())


def _build_synthetic_dataset(
    dataset_dir: Path,
    split_shard_sizes: dict[str, list[int]],
    seed: int = 0,
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    meta_dtype = np.dtype([("sample_id", "<i8")])
    splits: dict[str, dict] = {}
    sample_counter = 0

    for split, shard_sizes in split_shard_sizes.items():
        shard_entries = []
        for shard_index, n in enumerate(shard_sizes):
            x = rng.standard_normal((n, X_DIM)).astype(np.float32)
            # Make Y a smooth function of X so metrics are meaningful, plus noise.
            base = x @ rng.standard_normal((X_DIM, NUM_REGIONS * Y_NODE_WIDTH)).astype(
                np.float32
            )
            y = (base.reshape(n, NUM_REGIONS, Y_NODE_WIDTH) + 0.01 * rng.standard_normal(
                (n, NUM_REGIONS, Y_NODE_WIDTH)
            )).astype(np.float32)
            meta = np.zeros((n,), dtype=meta_dtype)
            meta["sample_id"] = np.arange(sample_counter, sample_counter + n)
            sample_counter += n

            x_name = f"{split}_{shard_index}_x.npy"
            y_name = f"{split}_{shard_index}_y.npy"
            m_name = f"{split}_{shard_index}_meta.npy"
            np.save(dataset_dir / x_name, x)
            np.save(dataset_dir / y_name, y)
            np.save(dataset_dir / m_name, meta)
            shard_entries.append(
                {"x": x_name, "y": y_name, "metadata": m_name, "n_samples": int(n)}
            )
        splits[split] = {"shards": shard_entries}

    manifest = {
        "storage": {
            "format": "npy_memmap_shards",
            "x_sample_shape": [X_DIM],
            "y_sample_shape": [NUM_REGIONS, Y_NODE_WIDTH],
        },
        "contract": {
            "y_node_width": Y_NODE_WIDTH,
            "targets": {
                "delay": {"y_columns": [0]},
                "cancel": {"y_columns": [1]},
                "infra": {"y_columns": [2, 3]},
            },
        },
        "splits": splits,
    }
    with (dataset_dir / "manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return dataset_dir


def _model_config(family: str) -> ModelTrainingConfig:
    if family == "MLP":
        return ModelTrainingConfig(
            family="MLP",
            profile="test_mlp",
            parameters=MLPProfileConfig(hidden_dims=[16, 8], dropout=0.0),
        )
    if family == "Auto":
        return ModelTrainingConfig(
            family="Auto",
            profile="test_auto",
            parameters=AutoProfileConfig(hidden_dims=[16, 8], latent_dim=4, dropout=0.0),
        )
    if family == "VAE":
        return ModelTrainingConfig(
            family="VAE",
            profile="test_vae",
            parameters=VAEProfileConfig(
                encoder_hidden_dims=[16, 8],
                decoder_hidden_dims=[8, 16],
                latent_dim=4,
                dropout=0.0,
                beta_warmup_epochs=1,
            ),
        )
    if family == "GP":
        return ModelTrainingConfig(
            family="GP",
            profile="test_gp",
            parameters=GPProfileConfig(
                num_inducing_points=8,
                num_latents=2,
                kernel_type="rbf",
                use_ard=False,
                eval_batch_size=32,
            ),
        )
    raise ValueError(family)


def _make_config(
    dataset_dir: Path,
    output_root: Path,
    family: str,
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
        model=_model_config(family),
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


def test_head_layout_and_slicing() -> None:
    """The multi-head Y equals per-head single-head outputs concatenated."""
    from src.data.dataset import create_dataset

    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = _build_synthetic_dataset(
            Path(tmp) / "data", {"train": [20, 12]}
        )

        for mode in ("whole", "chunked"):
            multi = create_dataset(
                dataset_dir, split="train", mode=mode, components=COMPONENTS
            )
            assert multi.head_slices == EXPECTED_SLICES, (mode, multi.head_slices)
            assert multi.head_output_dims == EXPECTED_WIDTHS, mode

            singles = {
                name: create_dataset(
                    dataset_dir, split="train", mode=mode, component=name
                )
                for name in COMPONENTS
            }

            for index in range(len(multi)):
                y_multi = multi[index]["y"].numpy().reshape(-1)
                assert y_multi.shape[0] == TOTAL_WIDTH
                for name, (start, stop) in EXPECTED_SLICES.items():
                    y_single = singles[name][index]["y"].numpy().reshape(-1)
                    np.testing.assert_allclose(
                        y_multi[start:stop], y_single, rtol=1e-6, atol=1e-6,
                        err_msg=f"mode={mode} head={name} index={index}",
                    )
    print("PASS test_head_layout_and_slicing")


def _run_multi(family: str, loader_mode: str) -> None:
    from src.models.multihead import MultiHeadSurrogate
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
            family=family,
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
            assert np.isfinite(head["MSE"]), (family, name)
            # Per-head Values = n_val_samples * head_width.
            assert head["Values"] == 24 * EXPECTED_WIDTHS[name], (family, name)
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
        assert len(table) == len(COMPONENTS), family
        assert set(table["Component"]) == set(COMPONENTS)
        assert set(table["Mode"]) == {"multi"}

        # Validation/Test/Training Time are separate columns, and each split's
        # per-head metrics carry the policy rank metrics.
        assert {"Validation", "Test", "Training Time (s)"}.issubset(table.columns)
        for name in COMPONENTS:
            assert "Policy Kendall Tau" in validation["Per Head"][name], (family, name)
            assert "Policy Aggregate Kendall Tau" in validation["Per Head"][name]
        row_test = table.iloc[0]["Test"]
        assert "Policy Kendall Tau" in row_test
        assert "Policy Aggregate Kendall Tau" in row_test

        # Saved checkpoint round-trips with matching head metadata.
        head_output_dims = {name: EXPECTED_WIDTHS[name] for name in COMPONENTS}
        reloaded = MultiHeadSurrogate(
            family=family,
            input_dim=X_DIM,
            head_output_dims=head_output_dims,
            head_slices=EXPECTED_SLICES,
            profile=config.model.parameters,
        )
        metadata = reloaded.load(result["run_paths"].model_path)
        assert metadata["Heads"] == head_output_dims
        assert reloaded.output_dim == TOTAL_WIDTH
    print(f"PASS _run_multi family={family} loader_mode={loader_mode}")


def test_multi_mlp_whole() -> None:
    _run_multi("MLP", "whole")


def test_multi_mlp_chunked() -> None:
    _run_multi("MLP", "chunked")


def test_multi_auto_whole() -> None:
    _run_multi("Auto", "whole")


def test_multi_vae_whole() -> None:
    _run_multi("VAE", "whole")


def test_multi_gp_whole() -> None:
    """Multi-head GP is a single shared LMC; heads are task-column groups.

    Skipped when ``gpytorch`` is unavailable (e.g. the local CPU venv). Inside the
    training container the full path runs end to end.
    """
    try:
        import gpytorch  # noqa: F401
    except ImportError:
        print("SKIP test_multi_gp_whole (gpytorch not installed)")
        return

    from src.models.gp import GPSurrogate
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
            family="GP",
            mode="multi",
            loader_mode="whole",
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

        run_yaml = next((output_root / "results" / "runs").glob("*.yaml"))
        record = yaml.safe_load(run_yaml.read_text(encoding="utf-8"))
        assert record["Mode"] == "multi"
        assert record["Component"] == "multi_all"
        assert record["Results"]["Heads"] == EXPECTED_WIDTHS

        import pandas as pd

        table = pd.read_pickle(output_root / "results" / "tables" / "run_history.pkl")
        assert len(table) == len(COMPONENTS)
        assert set(table["Component"]) == set(COMPONENTS)
        assert set(table["Mode"]) == {"multi"}

        # Round-trip the single shared LMC and confirm the head partition survives.
        head_output_dims = {name: EXPECTED_WIDTHS[name] for name in COMPONENTS}
        reloaded = GPSurrogate(
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
    print("PASS test_multi_gp_whole")


def test_single_head_unchanged() -> None:
    """The single-head path still produces flat metrics and one history row."""
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
            family="MLP",
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
    print("PASS test_single_head_unchanged")


def main() -> int:
    test_head_layout_and_slicing()
    test_multi_mlp_whole()
    test_multi_mlp_chunked()
    test_multi_auto_whole()
    test_multi_vae_whole()
    test_multi_gp_whole()
    test_single_head_unchanged()
    print("\nALL MULTI-HEAD SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
