"""Smoke tests for dataset analysis and prediction plotting.

Fabricates a tiny per-region ``npy_memmap_shards`` dataset (with the real contract
layout: year/rain/zone_water_depth/zone_modifiers + action one-hot in x, and one
mode-component + one region-component target in y), then:

* runs ``src.analyze_dataset`` and asserts the stats table, histograms, and
  water-depth point plots are produced (with and without zones);
* trains a tiny MLP, saves a checkpoint at the layout path, runs
  ``src.plot_predictions`` and asserts prediction-vs-true plots are produced.

CPU-only; needs ``pydantic>=2``, ``torch``, ``matplotlib``, ``numpy``.

Run directly:  python -m tests.test_analysis_smoke
Or with pytest: pytest tests/test_analysis_smoke.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

NUM_REGIONS = 3
MODES = ["car", "bicycle", "on_foot"]
ACTION_CLASSES = 2
# x per-node layout: year(1) rain(1) zone_water_depth(3) zone_modifiers(3) onehot(2) = 10
X_NODE_WIDTH = 10
# y per-node layout: delay[car,bicycle,on_foot] (0,1,2) + action (3) = 4
Y_NODE_WIDTH = 4
ZONE_IDS = [900, 901, 902]


def _build_dataset(dataset_dir: Path, split_sizes: dict[str, list[int]], seed: int = 0) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    meta_dtype = np.dtype([("sample_id", "<i8")])
    splits: dict[str, dict] = {}
    counter = 0

    for split, shard_sizes in split_sizes.items():
        entries = []
        for shard_index, n in enumerate(shard_sizes):
            x = rng.standard_normal((n, NUM_REGIONS, X_NODE_WIDTH)).astype(np.float32)
            # Make targets depend on water depth (cols 2:5) so plots are meaningful.
            water = x[:, :, 2:5]
            y = np.empty((n, NUM_REGIONS, Y_NODE_WIDTH), dtype=np.float32)
            y[:, :, 0:3] = 2.0 * water + 0.1 * rng.standard_normal((n, NUM_REGIONS, 3))
            y[:, :, 3] = water.mean(axis=2) + 0.1 * rng.standard_normal((n, NUM_REGIONS))
            meta = np.zeros((n,), dtype=meta_dtype)
            meta["sample_id"] = np.arange(counter, counter + n)
            counter += n

            names = (f"{split}_{shard_index}_x.npy", f"{split}_{shard_index}_y.npy", f"{split}_{shard_index}_m.npy")
            np.save(dataset_dir / names[0], x)
            np.save(dataset_dir / names[1], y)
            np.save(dataset_dir / names[2], meta)
            entries.append(
                {"x": names[0], "y": names[1], "metadata": names[2], "n_samples": int(n)}
            )
        splits[split] = {"shards": entries}

    features = [
        {"name": "year", "length": 1, "shape": [1], "include_in_x": True, "include_in_y": False},
        {"name": "rain", "length": 1, "shape": [1], "include_in_x": True, "include_in_y": False},
        {
            "name": "zone_water_depth",
            "length": NUM_REGIONS * len(MODES),
            "shape": [len(MODES), NUM_REGIONS],
            "include_in_x": True,
            "include_in_y": False,
        },
        {
            "name": "zone_modifiers",
            "length": NUM_REGIONS * len(MODES) * (ACTION_CLASSES - 1),
            "shape": [len(MODES), NUM_REGIONS, ACTION_CLASSES - 1],
            "include_in_x": True,
            "include_in_y": False,
        },
    ]
    manifest = {
        "dataset": {"name": dataset_dir.name, "objective": "Econ"},
        "storage": {
            "format": "npy_memmap_shards",
            "x_sample_shape": [NUM_REGIONS, X_NODE_WIDTH],
            "y_sample_shape": [NUM_REGIONS, Y_NODE_WIDTH],
        },
        "contract": {
            "num_regions": NUM_REGIONS,
            "modes": MODES,
            "action_classes": ACTION_CLASSES,
            "include_action_onehot": True,
            "x_node_width": X_NODE_WIDTH,
            "y_node_width": Y_NODE_WIDTH,
            "features": features,
            "targets": {
                "delay": {"y_columns": [0, 1, 2], "type": "mode_component", "modes": MODES},
                "action": {"y_columns": [3], "type": "region_component", "modes": None},
            },
        },
        "splits": splits,
    }
    with (dataset_dir / "manifest.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)


def _write_region_graph(project_root: Path) -> None:
    graph_dir = project_root / "configs" / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_nodes": NUM_REGIONS,
        "nodes": [{"id": zid, "name": f"Zone {zid}"} for zid in ZONE_IDS],
        "edges": {"source": [900, 901], "target": [901, 902]},
    }
    with (graph_dir / "region_graph.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _run_analyze(env: dict[str, str]) -> None:
    import importlib
    import os

    old = dict(os.environ)
    os.environ.update(env)
    try:
        module = importlib.import_module("src.analyze_dataset")
        importlib.reload(module)
        module.main()
    finally:
        os.environ.clear()
        os.environ.update(old)


def _test_analyze() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)
        dataset = "econ_synth"
        dataset_dir = project_root / "data" / dataset
        _build_dataset(dataset_dir, {"train": [20, 20], "val": [15], "test": [15]})
        _write_region_graph(project_root)

        _run_analyze(
            {
                "DATASET": dataset,
                "PROJECT_ROOT": str(project_root),
                "SPLITS": "train,val,test",
                "ZONES": "900,902",
                "MODE_DECIMALS": "2",
                "HIST_BINS": "20",
            }
        )

        for split in ("train", "val", "test"):
            out = project_root / "results" / "analysis" / "dataset" / dataset / split
            assert (out / "summary_statistics.csv").is_file(), f"missing summary for {split}"
            assert (out / "summary_statistics.txt").is_file()
            assert (out / "hist_input_zone_water_depth_car_.png").is_file() or any(
                out.glob("hist_input_*.png")
            ), "no input histograms"
            assert any(out.glob("hist_target_*.png")), "no target histograms"
            assert (out / "point_delay_aggregate_meanzones.png").is_file(), "no aggregate plot"
            assert any(out.glob("point_delay_*_meanzones.png")), "no per-mode plot"
            assert (out / "point_delay_car_zones.png").is_file(), "no per-zone plot"
            assert (out / "point_action_zones.png").is_file(), "no region per-zone plot"

        # Verify the stats table content: mode-component target expands to 3 series.
        import csv

        rows = list(csv.DictReader((project_root / "results" / "analysis" / "dataset" / dataset / "train" / "summary_statistics.csv").open()))
        series = {row["series"] for row in rows}
        assert "delay[car]" in series and "action" in series
        assert "zone_water_depth[on_foot]" in series
    print("analyze_dataset smoke: PASS")


def _test_analysis_common() -> None:
    from src.data.dataset import DatasetManifest
    from src.eval import analysis_common as ac

    with tempfile.TemporaryDirectory() as tmp:
        dataset_dir = Path(tmp) / "econ_synth"
        _build_dataset(dataset_dir, {"train": [10], "val": [10], "test": [10]})
        payload = DatasetManifest(dataset_dir).payload
        features = ac.reconstruct_x_feature_columns(payload)
        by_name = {name: (start, width, modes) for name, start, width, modes in features}
        assert by_name["year"] == (0, 1, None)
        assert by_name["rain"] == (1, 1, None)
        assert by_name["zone_water_depth"][:2] == (2, 3)
        assert by_name["zone_water_depth"][2] == MODES
        assert by_name["zone_modifiers"][:2] == (5, 3)
        assert by_name["action_onehot"][:2] == (8, 2)
        targets = {name: (cols, modes, ttype) for name, cols, modes, ttype in ac.target_layout(payload)}
        assert targets["delay"][0] == [0, 1, 2]
        assert targets["delay"][1] == MODES
        assert targets["action"][1] is None
        assert ac.rounded_mode(np.array([1.111, 1.114, 2.5]), 2) == 1.11
    print("analysis_common smoke: PASS")


def _test_plot_predictions() -> None:
    import importlib
    import os

    import torch

    from src.config.schema import (
        MLPProfileConfig,
        ModelTrainingConfig,
        TrainLoopConfig,
    )
    from src.data.training import create_training_dataset, create_training_loader
    from src.models.registry import create_surrogate_model

    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)
        train_dataset = "econ_synth"
        dataset_dir = project_root / "data" / train_dataset
        _build_dataset(dataset_dir, {"train": [40], "val": [20], "test": [20]})
        _write_region_graph(project_root)

        component = "delay"
        family, profile, source_job = "MLP", "test_mlp", "job123"
        model_config = ModelTrainingConfig(
            family=family,
            profile=profile,
            parameters=MLPProfileConfig(hidden_dims=[8], dropout=0.0),
        )
        train_ds = create_training_dataset(dataset_dir, "train", "whole", component)
        val_ds = create_training_dataset(dataset_dir, "val", "whole", component)
        train_loader = create_training_loader(train_ds, 16, True, 0, False)
        val_loader = create_training_loader(val_ds, 16, False, 0, False)
        input_dim = train_ds[0][0].numel()
        output_dim = train_ds[0][1].numel()

        model = create_surrogate_model(model_config, input_dim, output_dim)
        checkpoint_path = (
            project_root
            / "outputs"
            / family
            / "single"
            / train_dataset
            / component
            / profile
            / source_job
            / "checkpoints"
            / "model.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        model.fit(train_loader, val_loader, TrainLoopConfig(epochs=2, print_every=1), checkpoint_path)
        model.save(checkpoint_path)
        assert checkpoint_path.is_file()

        env = {
            "OBJECTIVE": "Econ",
            "TRAIN_DATASET": train_dataset,
            "MODE": "single",
            "MODEL": family,
            "PARAMETERS": profile,
            "COMPONENT": component,
            "SOURCE_JOB_ID": source_job,
            "SPLIT": "val",
            "ZONES": "900,902",
            "LOADER_MODE": "whole",
            "PROJECT_ROOT": str(project_root),
        }
        old = dict(os.environ)
        os.environ.update(env)
        try:
            module = importlib.import_module("src.plot_predictions")
            importlib.reload(module)
            module.main()
        finally:
            os.environ.clear()
            os.environ.update(old)

        out = (
            project_root
            / "results"
            / "analysis"
            / "predictions"
            / family
            / "single"
            / train_dataset
            / component
            / profile
            / source_job
            / "val"
        )
        assert any(out.glob("pred_delay_*_meanzones.png")), "no per-mode pred plot"
        assert (out / "pred_delay_aggregate_meanzones.png").is_file(), "no aggregate pred plot"
        assert (out / "pred_delay_car_zones.png").is_file(), "no per-zone pred plot"
    print("plot_predictions smoke: PASS")


def main() -> None:
    _test_analysis_common()
    _test_analyze()
    _test_plot_predictions()
    print("\nAll analysis smoke tests passed.")


def test_analysis_common() -> None:
    _test_analysis_common()


def test_analyze_dataset() -> None:
    _test_analyze()


def test_plot_predictions() -> None:
    _test_plot_predictions()


if __name__ == "__main__":
    main()
