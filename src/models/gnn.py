"""Graph neural network surrogate over the region adjacency graph.

Every sample is stored per-region and flattened region-major by the training
layer, so a flat input vector of length ``num_regions * x_node_width`` reshapes to
``[num_regions, x_node_width]`` node features and a flat target of length
``num_regions * y_width`` reshapes to ``[num_regions, y_width]`` node targets. This
surrogate reshapes the flat vector, runs message passing over the region graph
(built from ``graph_adj.txt`` / ``node_mapping.txt`` / ``geodata_dict.txt``), and
flattens the per-node predictions back to the target layout. Because the model
consumes and produces the same flat ``[B, D]`` tensors as the MLP, it reuses the
shared training loop, standardizer, and metrics unchanged.

The graph is dependency-free (no torch_geometric): message passing is a dense
``[N, N]`` matmul, which is inexpensive for the ~278 regions in this project. The
edge list is persisted in the checkpoint so a saved model reloads without the
adjacency text files.

Single vs multi-head is handled in one class (mirroring the GP/XGB families): a
shared graph trunk feeds either one node-wise output head (single) or one
node-wise head per component whose flattened blocks are concatenated in head
order (multi), so each head reproduces its single-head output exactly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch.utils.data import DataLoader

from src.config.schema import GNNProfileConfig, TrainLoopConfig
from src.data.graph import (
    RegionGraph,
    build_gcn_operator,
    build_mean_operator,
    load_region_graph,
)
from src.eval.metrics import evaluate_regression, evaluate_regression_multihead
from src.models.base import SurrogateModel
from src.models.mlp import create_activation
from src.train.loop import fit_torch_model, predict_torch_model, select_device
from src.train.transform import TensorStandardizer


class GraphConvLayer(torch.nn.Module):
    """One message-passing layer over a fixed normalized operator.

    ``gcn`` applies ``linear(operator @ h)`` (the operator already includes
    self-loops and symmetric normalization). ``sage`` keeps the self and neighbour
    transforms separate: ``linear_self(h) + linear_neigh(operator @ h)`` where the
    operator is a row-normalized neighbour mean.
    """

    def __init__(self, in_dim: int, out_dim: int, conv_type: str) -> None:
        super().__init__()
        self.conv_type = str(conv_type)
        if self.conv_type == "gcn":
            self.linear = torch.nn.Linear(in_dim, out_dim)
        elif self.conv_type == "sage":
            self.linear_self = torch.nn.Linear(in_dim, out_dim)
            self.linear_neighbour = torch.nn.Linear(in_dim, out_dim, bias=False)
        else:
            raise ValueError(f"Unknown conv_type={self.conv_type!r}; expected gcn or sage")

    def forward(self, node_features: torch.Tensor, operator: torch.Tensor) -> torch.Tensor:
        aggregated = torch.einsum("ij,bjf->bif", operator, node_features)
        if self.conv_type == "gcn":
            return self.linear(aggregated)
        return self.linear_self(node_features) + self.linear_neighbour(aggregated)


class GNNNetwork(torch.nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        node_in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        activation: str,
        conv_type: str,
        residual: bool,
        layer_norm: bool,
        operator: torch.Tensor,
        single_out_dim: int | None = None,
        head_layout: list[tuple[str, int]] | None = None,
    ) -> None:
        super().__init__()
        if single_out_dim is None and head_layout is None:
            raise ValueError("Provide either single_out_dim or head_layout")
        self.num_nodes = int(num_nodes)
        self.node_in_dim = int(node_in_dim)
        # The operator depends only on the (fixed) graph, so it is rebuilt from the
        # persisted edge list on load and kept out of the state dict.
        self.register_buffer("operator", operator, persistent=False)

        self.encoder = torch.nn.Linear(self.node_in_dim, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [GraphConvLayer(hidden_dim, hidden_dim, conv_type) for _ in range(int(num_layers))]
        )
        self.norms = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(hidden_dim) if layer_norm else torch.nn.Identity()
                for _ in range(int(num_layers))
            ]
        )
        self.activation = create_activation(activation)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        self.residual = bool(residual)

        if head_layout is not None:
            self.head_order: list[str] | None = [name for name, _ in head_layout]
            self.decoders = torch.nn.ModuleDict(
                {name: torch.nn.Linear(hidden_dim, int(width)) for name, width in head_layout}
            )
        else:
            self.head_order = None
            self.decoder = torch.nn.Linear(hidden_dim, int(single_out_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        node_features = values.reshape(batch, self.num_nodes, self.node_in_dim)
        hidden = self.activation(self.encoder(node_features))
        for layer, norm in zip(self.layers, self.norms):
            updated = self.dropout(self.activation(layer(hidden, self.operator)))
            if self.residual:
                updated = updated + hidden
            hidden = norm(updated)

        if self.head_order is not None:
            blocks = [self.decoders[name](hidden).reshape(batch, -1) for name in self.head_order]
            return torch.cat(blocks, dim=1)
        return self.decoder(hidden).reshape(batch, -1)


class GNNSurrogate(SurrogateModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        profile: GNNProfileConfig,
        graph_search_roots: list[Path] | None = None,
        head_slices: Mapping[str, tuple[int, int]] | None = None,
        head_output_dims: Mapping[str, int] | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.profile = profile
        self.graph_search_roots = (
            [Path(root) for root in graph_search_roots]
            if graph_search_roots is not None
            else None
        )
        # Multi-head metadata. When set, this GNN shares one graph trunk and emits
        # one node-wise head per component; ``head_slices`` partition the flattened
        # concatenation so metrics/results are reported per component, exactly like
        # the shared-trunk families. ``None`` keeps the single-head behaviour.
        self.head_slices: dict[str, tuple[int, int]] | None = (
            {str(name): (int(bounds[0]), int(bounds[1])) for name, bounds in head_slices.items()}
            if head_slices is not None
            else None
        )
        if head_output_dims is not None:
            self.head_output_dims: dict[str, int] | None = {
                str(name): int(dim) for name, dim in head_output_dims.items()
            }
        elif self.head_slices is not None:
            self.head_output_dims = {
                name: stop - start for name, (start, stop) in self.head_slices.items()
            }
        else:
            self.head_output_dims = None

        self.num_nodes: int | None = None
        self.region_ids: list[int] | None = None
        self._edge_index: torch.Tensor | None = None
        self.network: GNNNetwork | None = None
        self.standardizer: TensorStandardizer | None = None
        self.device = "cpu"

        # Training constructs the model with search roots, so the graph is loaded
        # (and mismatches surface) immediately. Loading a saved model can pass no
        # roots; ``load`` then rebuilds the graph from the persisted edge list.
        if self.graph_search_roots is not None:
            graph = load_region_graph(
                graph_file=self.profile.graph_file,
                search_roots=self.graph_search_roots,
            )
            self._build_network(graph.edge_index, graph.num_nodes, graph.region_ids)

    def _build_network(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        region_ids: list[int] | None = None,
    ) -> None:
        num_nodes = int(num_nodes)
        if self.input_dim % num_nodes != 0:
            raise ValueError(
                f"input_dim={self.input_dim} is not divisible by num_nodes={num_nodes}; "
                "the flat input must reshape to [num_nodes, x_node_width]."
            )
        if self.output_dim % num_nodes != 0:
            raise ValueError(
                f"output_dim={self.output_dim} is not divisible by num_nodes={num_nodes}; "
                "the flat target must reshape to [num_nodes, y_width]."
            )

        if self.profile.conv_type == "gcn":
            operator = build_gcn_operator(
                edge_index,
                num_nodes,
                undirected=self.profile.undirected,
                add_self_loops=self.profile.add_self_loops,
            )
        else:
            operator = build_mean_operator(
                edge_index, num_nodes, undirected=self.profile.undirected
            )

        node_in_dim = self.input_dim // num_nodes
        head_layout: list[tuple[str, int]] | None = None
        single_out_dim: int | None = None
        if self.head_output_dims is not None:
            head_layout = []
            for name, width in self.head_output_dims.items():
                if width % num_nodes != 0:
                    raise ValueError(
                        f"Head {name!r} width={width} is not divisible by num_nodes={num_nodes}."
                    )
                head_layout.append((name, width // num_nodes))
        else:
            single_out_dim = self.output_dim // num_nodes

        self.network = GNNNetwork(
            num_nodes=num_nodes,
            node_in_dim=node_in_dim,
            hidden_dim=self.profile.hidden_dim,
            num_layers=self.profile.num_layers,
            dropout=self.profile.dropout,
            activation=self.profile.activation,
            conv_type=self.profile.conv_type,
            residual=self.profile.residual,
            layer_norm=self.profile.layer_norm,
            operator=operator,
            single_out_dim=single_out_dim,
            head_layout=head_layout,
        )
        self.num_nodes = num_nodes
        self.region_ids = list(region_ids) if region_ids is not None else None
        self._edge_index = edge_index.detach().to(torch.long).cpu()

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        config: TrainLoopConfig,
        checkpoint_path: str | Path,
        periodic_save_callback: Callable[[int], None] | None = None,
    ) -> dict[str, Any]:
        if self.network is None:
            raise RuntimeError(
                "The region graph was not loaded; construct the model with "
                "graph_search_roots pointing at the adjacency files before fitting."
            )
        self.standardizer = TensorStandardizer(
            standardize_inputs=config.standardization.inputs,
            standardize_targets=config.standardization.targets,
            epsilon=config.standardization.epsilon,
        )
        print("[training] fitting standardization statistics", flush=True)
        training_dataset = getattr(train_loader, "dataset", None)
        materialized = (
            training_dataset.materialized_tensors()
            if hasattr(training_dataset, "materialized_tensors")
            else None
        )
        if materialized is not None:
            self.standardizer.fit_from_tensors(materialized[0], materialized[1])
        else:
            self.standardizer.fit(train_loader)
        training_result = fit_torch_model(
            model=self.network,
            train_loader=train_loader,
            validation_loader=validation_loader,
            standardizer=self.standardizer,
            config=config,
            checkpoint_path=checkpoint_path,
            periodic_save_callback=periodic_save_callback,
        )
        self.device = str(training_result["Device"])
        return training_result

    def predict(
        self,
        loader: DataLoader,
        max_batches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.standardizer is None or self.network is None:
            raise RuntimeError("The model must be fitted or loaded before prediction")
        self.network.to(self.device)
        self.standardizer.move(self.device)
        return predict_torch_model(
            model=self.network,
            loader=loader,
            standardizer=self.standardizer,
            device=self.device,
            max_batches=max_batches,
        )

    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        if self.standardizer is None or self.network is None:
            raise RuntimeError("The model must be fitted or loaded before evaluation")
        self.network.to(self.device)
        self.standardizer.move(self.device)
        if self.head_slices is not None:
            return evaluate_regression_multihead(
                model=self.network,
                loader=loader,
                device=self.device,
                standardizer=self.standardizer,
                head_slices=self.head_slices,
            )
        return evaluate_regression(
            model=self.network,
            loader=loader,
            device=self.device,
            standardizer=self.standardizer,
        )

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        if self.standardizer is None or self.network is None or self._edge_index is None:
            raise RuntimeError("The model must be fitted or loaded before saving")
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        is_multi = self.head_slices is not None
        torch.save(
            {
                "family": "GNN",
                "mode": "multi" if is_multi else "single",
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "num_nodes": int(self.num_nodes or 0),
                "edge_index": self._edge_index,
                "region_ids": list(self.region_ids) if self.region_ids is not None else None,
                "profile": self.profile.model_dump(mode="python"),
                "head_output_dims": (
                    dict(self.head_output_dims) if self.head_output_dims is not None else None
                ),
                "head_slices": (
                    {name: list(bounds) for name, bounds in self.head_slices.items()}
                    if is_multi
                    else None
                ),
                "model_state": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.network.state_dict().items()
                },
                "standardizer_state": self.standardizer.state_dict(),
                "metadata": metadata or {},
            },
            output_path,
        )

    def load(self, path: str | Path) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
        if int(checkpoint["input_dim"]) != self.input_dim:
            raise ValueError(
                f"Checkpoint input_dim={checkpoint['input_dim']} does not match "
                f"model input_dim={self.input_dim}"
            )
        if int(checkpoint["output_dim"]) != self.output_dim:
            raise ValueError(
                f"Checkpoint output_dim={checkpoint['output_dim']} does not match "
                f"model output_dim={self.output_dim}"
            )
        head_slices = checkpoint.get("head_slices")
        if head_slices is not None:
            self.head_slices = {
                str(name): (int(bounds[0]), int(bounds[1]))
                for name, bounds in head_slices.items()
            }
        head_output_dims = checkpoint.get("head_output_dims")
        if head_output_dims is not None:
            self.head_output_dims = {str(name): int(dim) for name, dim in head_output_dims.items()}

        edge_index = torch.as_tensor(checkpoint["edge_index"], dtype=torch.long)
        num_nodes = int(checkpoint["num_nodes"])
        region_ids = checkpoint.get("region_ids")
        self._build_network(edge_index, num_nodes, region_ids)
        assert self.network is not None
        self.network.load_state_dict(checkpoint["model_state"])

        standardizer_state = checkpoint["standardizer_state"]
        self.standardizer = TensorStandardizer(
            standardize_inputs=bool(standardizer_state["standardize_inputs"]),
            standardize_targets=bool(standardizer_state["standardize_targets"]),
            epsilon=float(standardizer_state["epsilon"]),
        )
        self.standardizer.load_state_dict(standardizer_state)
        self.device = select_device("auto")
        return dict(checkpoint.get("metadata", {}))
