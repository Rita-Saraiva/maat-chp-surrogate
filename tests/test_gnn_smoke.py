"""End-to-end smoke tests for the GNN surrogate on a synthetic dataset.

These fabricate a tiny ``npy_memmap_shards`` dataset whose samples are stored
per-region (``x[num_nodes, x_node_width]`` / ``y[num_nodes, y_width]``) plus a
matching temp region graph (the consolidated ``configs/graph/region_graph.yaml``
plus the legacy ``graph_adj.txt`` / ``node_mapping.txt`` / ``geodata_dict.txt``
sources), then exercise the full GNN path (config -> data layer ->
graph message passing -> per-head metrics -> results store) for single- and
multi-head runs and both conv types, plus a save/load round-trip that must
reproduce the fitted predictions and reload without the adjacency files.

They run CPU-only and need ``pydantic>=2`` and ``torch``.

Run directly:  python -m tests.test_gnn_smoke
Or with pytest: pytest tests/test_gnn_smoke.py
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
    GNNProfileConfig,
    ModelTrainingConfig,
    StandardizationConfig,
    TrainingRunConfig,
    TrainLoopConfig,
)

NUM_NODES = 5
X_NODE_WIDTH = 3
X_DIM = NUM_NODES * X_NODE_WIDTH
Y_NODE_WIDTH = 4
COMPONENTS = ["delay", "cancel", "infra"]
# delay -> col [0], cancel -> col [1], infra -> cols [2, 3]; each head flattens
# region-major, so its width is num_nodes * len(columns).
EXPECTED_WIDTHS = {"delay": NUM_NODES, "cancel": NUM_NODES, "infra": 2 * NUM_NODES}
EXPECTED_SLICES = {
    "delay": (0, NUM_NODES),
    "cancel": (NUM_NODES, 2 * NUM_NODES),
    "infra": (2 * NUM_NODES, 4 * NUM_NODES),
}
TOTAL_WIDTH = sum(EXPECTED_WIDTHS.values())
REGION_IDS = [1001, 1002, 1003, 1004, 1005]
# Ring graph over the five nodes (directed edges; the loader symmetrizes).
RING_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]


def _write_graph_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    sources = [REGION_IDS[a] for a, _ in RING_EDGES]
    destinations = [REGION_IDS[b] for _, b in RING_EDGES]
    (root / "graph_adj.txt").write_text(
        " ".join(str(value) for value in sources)
        + "\n"
        + " ".join(str(value) for value in destinations)
        + "\n",
        encoding="utf-8",
    )
    mapping_lines = ["# Node mapping: region_id -> node_index"]
    mapping_lines += [f"{region}: {index}" for index, region in enumerate(REGION_IDS)]
    (root / "node_mapping.txt").write_text("\n".join(mapping_lines) + "\n", encoding="utf-8")
    geodata_lines = [f"{region}: Region {index}" for index, region in enumerate(REGION_IDS)]
    (root / "geodata_dict.txt").write_text("\n".join(geodata_lines) + "\n", encoding="utf-8")

    # Consolidated YAML at the default graph_file location, produced by the same
    # converter the project ships, so the GNN's default graph source is exercised.
    from tools.convert_graph_to_yaml import build_yaml_text

    yaml_text = build_yaml_text(
        root, "graph_adj.txt", "node_mapping.txt", "geodata_dict.txt"
    )
    yaml_path = root / "configs" / "graph" / "region_graph.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_text, encoding="utf-8")


def _build_synthetic_dataset(
    dataset_dir: Path,
    split_shard_sizes: dict[str, list[int]],
    seed: int = 0,
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    meta_dtype = np.dtype([("sample_id", "<i8")])
    # Fixed projection so Y is a smooth function of the flattened per-node X.
    projection = rng.standard_normal((X_DIM, NUM_NODES * Y_NODE_WIDTH)).astype(np.float32)
    splits: dict[str, dict] = {}
    sample_counter = 0

    for split, shard_sizes in split_shard_sizes.items():
        shard_entries = []
        for shard_index, n in enumerate(shard_sizes):
            x = rng.standard_normal((n, NUM_NODES, X_NODE_WIDTH)).astype(np.float32)
            base = x.reshape(n, X_DIM) @ projection
            y = (
                base.reshape(n, NUM_NODES, Y_NODE_WIDTH)
                + 0.01 * rng.standard_normal((n, NUM_NODES, Y_NODE_WIDTH))
            ).astype(np.float32)
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
            "x_sample_shape": [NUM_NODES, X_NODE_WIDTH],
            "y_sample_shape": [NUM_NODES, Y_NODE_WIDTH],
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


def _model_config(conv_type: str = "gcn") -> ModelTrainingConfig:
    return ModelTrainingConfig(
        family="GNN",
        profile="test_gnn",
        parameters=GNNProfileConfig(
            hidden_dim=16,
            num_layers=2,
            dropout=0.0,
            activation="gelu",
            conv_type=conv_type,
            residual=True,
            layer_norm=True,
        ),
    )


def _make_config(
    dataset_dir: Path,
    output_root: Path,
    mode: str,
    loader_mode: str,
    component: str,
    components: list[str] | None,
    conv_type: str = "gcn",
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
        model=_model_config(conv_type),
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


def test_graph_loader() -> None:
    """The loader maps region ids to row-aligned node indices and builds operators."""
    import torch

    from src.data.graph import (
        build_gcn_operator,
        build_mean_operator,
        load_region_graph,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_graph_files(root)
        graph = load_region_graph(search_roots=[root])

        assert graph.num_nodes == NUM_NODES
        assert graph.region_ids == REGION_IDS
        assert graph.edge_index.shape == (2, len(RING_EDGES))
        # Node indices, not region ids.
        assert int(graph.edge_index.max()) < NUM_NODES

        gcn = build_gcn_operator(graph.edge_index, NUM_NODES)
        assert gcn.shape == (NUM_NODES, NUM_NODES)
        # Symmetric normalization is symmetric.
        assert torch.allclose(gcn, gcn.t(), atol=1e-6)
        # Self-loops populate the diagonal.
        assert bool((gcn.diagonal() > 0).all())

        mean = build_mean_operator(graph.edge_index, NUM_NODES)
        # Row-normalized neighbour mean: each row sums to 1 on this connected ring.
        assert torch.allclose(mean.sum(dim=1), torch.ones(NUM_NODES), atol=1e-6)
        # No self term in the mean operator.
        assert torch.allclose(mean.diagonal(), torch.zeros(NUM_NODES), atol=1e-6)
    print("PASS test_graph_loader")


def _run_single(loader_mode: str, conv_type: str) -> None:
    from src.train.trainer import train_surrogate

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_graph_files(root)
        dataset_dir = _build_synthetic_dataset(
            root / "data", {"train": [40], "val": [24], "test": [24]}
        )
        output_root = root / "out"
        config = _make_config(
            dataset_dir,
            output_root,
            mode="single",
            loader_mode=loader_mode,
            component="delay",
            components=None,
            conv_type=conv_type,
        )
        result = train_surrogate(config)

        validation = result["validation"]
        assert "MSE" in validation and "R2" in validation
        assert "Per Head" not in validation
        assert validation["Values"] == 24 * EXPECTED_WIDTHS["delay"]
        assert np.isfinite(validation["MSE"])

        import pandas as pd

        table = pd.read_pickle(output_root / "results" / "tables" / "run_history.pkl")
        assert len(table) == 1
        assert table.iloc[0]["Component"] == "delay"
        assert table.iloc[0]["Mode"] == "single"

        # Validation/Test/Training Time are separate columns, and the policy rank
        # metrics are attached to each split's metrics.
        assert {"Validation", "Test", "Training Time (s)"}.issubset(table.columns)
        assert "Policy Kendall Tau" in validation
        assert "Policy Aggregate Kendall Tau" in validation
        assert "Policy Kendall Tau" in table.iloc[0]["Test"]
    print(f"PASS _run_single loader_mode={loader_mode} conv_type={conv_type}")


def _run_multi(loader_mode: str, conv_type: str) -> None:
    from src.models.gnn import GNNSurrogate
    from src.train.trainer import train_surrogate

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_graph_files(root)
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
            conv_type=conv_type,
        )
        result = train_surrogate(config)

        validation = result["validation"]
        assert set(validation.keys()) == {"Overall", "Per Head"}
        assert set(validation["Per Head"].keys()) == set(COMPONENTS)
        for name in COMPONENTS:
            head = validation["Per Head"][name]
            assert np.isfinite(head["MSE"]), (conv_type, name)
            assert head["Values"] == 24 * EXPECTED_WIDTHS[name], (conv_type, name)
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

        # Round-trip the shared trunk + per-head decoders, reloading WITHOUT the
        # adjacency files (the edge list is persisted in the checkpoint).
        head_output_dims = {name: EXPECTED_WIDTHS[name] for name in COMPONENTS}
        reloaded = GNNSurrogate(
            input_dim=X_DIM,
            output_dim=TOTAL_WIDTH,
            profile=config.model.parameters,
            graph_search_roots=None,
            head_slices=EXPECTED_SLICES,
            head_output_dims=head_output_dims,
        )
        metadata = reloaded.load(result["run_paths"].model_path)
        assert metadata["Heads"] == head_output_dims
        assert reloaded.output_dim == TOTAL_WIDTH
        assert reloaded.head_slices == EXPECTED_SLICES
    print(f"PASS _run_multi loader_mode={loader_mode} conv_type={conv_type}")


def _loaders(dataset_dir: Path, component: str):
    from src.data.training import create_training_dataset, create_training_loader

    loaders = {}
    for split in ("train", "val"):
        dataset = create_training_dataset(
            dataset_dir=dataset_dir,
            split=split,
            loader_mode="whole",
            component=component,
            components=None,
        )
        loaders[split] = create_training_loader(
            dataset=dataset,
            batch_size=16,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
        )
    return loaders


def test_save_load_reproduces_predictions() -> None:
    """A reloaded GNN reproduces the fitted predictions bit-for-bit (eval mode)."""
    import torch

    from src.models.gnn import GNNSurrogate

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_graph_files(root)
        dataset_dir = _build_synthetic_dataset(
            root / "data", {"train": [40], "val": [24]}
        )
        loaders = _loaders(dataset_dir, component="delay")
        profile = _model_config().parameters
        output_dim = EXPECTED_WIDTHS["delay"]

        model = GNNSurrogate(
            input_dim=X_DIM,
            output_dim=output_dim,
            profile=profile,
            graph_search_roots=[root],
        )
        config = TrainLoopConfig(
            epochs=3,
            patience=3,
            print_every=1,
            seed=0,
            device="cpu",
            deterministic=True,
            standardization=StandardizationConfig(inputs=True, targets=True),
        )
        checkpoint_path = root / "best_state.pt"
        model.fit(loaders["train"], loaders["val"], config, checkpoint_path)
        predictions_before, _ = model.predict(loaders["val"])

        model_path = root / "model.pt"
        model.save(model_path, metadata={"Component": "delay"})

        reloaded = GNNSurrogate(
            input_dim=X_DIM,
            output_dim=output_dim,
            profile=profile,
            graph_search_roots=None,  # prove reload works without the txt files
        )
        reloaded.load(model_path)
        predictions_after, _ = reloaded.predict(loaders["val"])

        assert torch.allclose(predictions_before, predictions_after, rtol=1e-5, atol=1e-6)
    print("PASS test_save_load_reproduces_predictions")


def test_single_gcn_whole() -> None:
    _run_single("whole", "gcn")


def test_single_gcn_chunked() -> None:
    _run_single("chunked", "gcn")


def test_single_sage_whole() -> None:
    _run_single("whole", "sage")


def test_multi_gcn_whole() -> None:
    _run_multi("whole", "gcn")


def test_multi_gcn_chunked() -> None:
    _run_multi("chunked", "gcn")


def test_multi_sage_whole() -> None:
    _run_multi("whole", "sage")


def main() -> None:
    test_graph_loader()
    test_single_gcn_whole()
    test_single_gcn_chunked()
    test_single_sage_whole()
    test_multi_gcn_whole()
    test_multi_gcn_chunked()
    test_multi_sage_whole()
    test_save_load_reproduces_predictions()
    print("\nALL GNN SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
