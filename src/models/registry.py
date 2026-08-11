from __future__ import annotations

from pathlib import Path
from typing import Mapping

from src.config.schema import ModelTrainingConfig
from src.models.auto import AutoSurrogate
from src.models.base import SurrogateModel
from src.models.mlp import MLPSurrogate
from src.models.vae import VAESurrogate


def create_surrogate_model(
    config: ModelTrainingConfig,
    input_dim: int,
    output_dim: int,
    graph_search_roots: list[Path] | None = None,
) -> SurrogateModel:
    if config.family == "MLP":
        return MLPSurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            profile=config.parameters,
        )
    if config.family == "Auto":
        return AutoSurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            profile=config.parameters,
        )
    if config.family == "VAE":
        return VAESurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            profile=config.parameters,
        )
    if config.family == "GP":
        from src.models.gp import GPSurrogate

        return GPSurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            profile=config.parameters,
        )
    if config.family == "XGB":
        from src.models.xgboost import XGBSurrogate

        return XGBSurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            profile=config.parameters,
        )
    if config.family == "GNN":
        from src.models.gnn import GNNSurrogate

        return GNNSurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            profile=config.parameters,
            graph_search_roots=graph_search_roots,
        )
    raise ValueError(f"Unknown model family {config.family!r}")


def create_multihead_surrogate_model(
    config: ModelTrainingConfig,
    input_dim: int,
    head_output_dims: Mapping[str, int],
    head_slices: Mapping[str, tuple[int, int]],
    graph_search_roots: list[Path] | None = None,
) -> SurrogateModel:
    if config.family == "GP":
        # A multi-head GP is a single shared LMC over the concatenated task axis;
        # ``head_slices`` partition that axis so metrics/results are reported per
        # component, matching the shared-trunk families.
        from src.models.gp import GPSurrogate

        output_dim = sum(int(dim) for dim in head_output_dims.values())
        return GPSurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            profile=config.parameters,
            head_slices=head_slices,
            head_output_dims=head_output_dims,
        )

    if config.family == "XGB":
        # A multi-head XGBoost model is a single native multi-output regressor over
        # the concatenated task axis; ``head_slices`` partition that axis so
        # metrics/results are reported per component, matching the other families.
        from src.models.xgboost import XGBSurrogate

        output_dim = sum(int(dim) for dim in head_output_dims.values())
        return XGBSurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            profile=config.parameters,
            head_slices=head_slices,
            head_output_dims=head_output_dims,
        )

    if config.family == "GNN":
        # A multi-head GNN shares one graph trunk and emits one node-wise head per
        # component; ``head_slices`` partition the flattened concatenation so
        # metrics/results are reported per component, matching the other families.
        from src.models.gnn import GNNSurrogate

        output_dim = sum(int(dim) for dim in head_output_dims.values())
        return GNNSurrogate(
            input_dim=input_dim,
            output_dim=output_dim,
            profile=config.parameters,
            graph_search_roots=graph_search_roots,
            head_slices=head_slices,
            head_output_dims=head_output_dims,
        )

    from src.models.multihead import MultiHeadSurrogate

    return MultiHeadSurrogate(
        family=config.family,
        input_dim=input_dim,
        head_output_dims=head_output_dims,
        head_slices=head_slices,
        profile=config.parameters,
    )
