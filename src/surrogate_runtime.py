"""
Loads trained `src/` surrogate checkpoints and monkey-patches `maat.world` /
`maat.quality_of_life` at the class level so `BasicEnvironment.step()` uses them instead of
running the real simulator, for whichever reward components are configured.

The `maat` package itself (`RL_framework/maat/` in this repo's local reference copy; wherever your
job actually deploys it on the cluster) is never edited by this module -- everything here is applied from
the outside, once per process, before any `BasicEnvironment` is constructed. If nothing is
configured (every `env.surrogate.<target>` entry is null/absent), none of the patches change
behaviour: the econ-triad patch is only installed when `use_surrogate=True` is actually passed
to `BasicEnvironment`, and the qol/action/maintenance patches immediately delegate to the
original unpatched method when their target has no loaded surrogate.

Six canonical single-component targets are supported, matching
`src/data/contract.py::CANONICAL_REWARD_COMPONENTS` /
`src/config/load.py::TRAINING_OBJECTIVE_COMPONENTS`:

    Econ: delay, cancel, infra   (drive the transport step)
    QoL:  qol, action, maintenance

`delay`/`cancel`/`infra` are only used if *all three* are configured -- MAAT's transport
simulator computes them together in one call, so there is no way to substitute just one of the
three without running the (whole) real simulation for that step. `qol`, `action`, and
`maintenance` are each computed independently of the transport step and of each other, so they
are substituted independently.

NOTE: this file deliberately does NOT import `src.eval.cross_dataset` -- that module imports
`CrossEvalRecord` / `append_cross_eval_history` from `src.results.training_results`, which are
not defined there as of this writing, so importing `src.eval.cross_dataset` raises `ImportError`.
The (small) pieces of that module's logic we need -- the checkpoint path formula and the
checkpoint -> model-shell reconstruction -- are reimplemented directly below against
`src/models/registry.py` and `src/config/schema.py`, which do not have that problem.

INPUT SHAPE (important -- read before touching `build_surrogate_input`):
`src/data/assembly.py::observation_to_xy`/`feature_to_nodes` show that each trained surrogate
is a *per-region* model: one training row = one region's own slice of one RL step's observation
(`x.shape == (num_regions, x_node_width)`, `y.shape == (num_regions, y_node_width)`). So a single
inference call for one RL step is a *batch* of `num_regions` rows, not one flattened whole-scenario
vector -- `build_surrogate_input` below returns exactly that `(num_regions, D)` batch, and model
outputs are reshaped back into per-mode/per-region arrays accordingly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Canonical component metadata (mirrors src/data/contract.py / src/config/load.py)
# --------------------------------------------------------------------------- #

ECON_TARGETS = ("delay", "cancel", "infra")
QOL_TARGETS = ("qol",)
ACTION_TARGETS = ("action", "maintenance")
ALL_TARGETS = ECON_TARGETS + QOL_TARGETS + ACTION_TARGETS

# target_name -> maat.reward.RewardComponentName.<member>.value (the Impacts dict key)
TARGET_TO_IMPACTS_KEY = {
    "delay": "transport_indirect_delays",
    "cancel": "transport_indirect_notravel",
    "infra": "transport_direct_infrastructure",
    "qol": "wellbeing_quality_of_life",
    "action": "actions_direct_cost",
    "maintenance": "actions_maintenance_cost",
}

# target_name -> input_style, from src/data/contract.py::CANONICAL_REWARD_COMPONENTS.
# mode_depth / mode_infra keep {rain, zone_water_depth, zone_modifiers} (year + action one-hot
# dropped); region_full_graph keeps the full per-region node vector (year, rain,
# zone_water_depth, zone_modifiers, action one-hot). Multi-head checkpoints always use the full
# vector too (see LoadedSurrogate.full_width), regardless of a head's own input_style.
TARGET_INPUT_STYLE = {
    "delay": "mode_depth",
    "cancel": "mode_depth",
    "infra": "mode_infra",
    "qol": "mode_depth",
    "action": "region_full_graph",
    "maintenance": "region_full_graph",
}

MULTI_COMPONENT_LABEL = "multi_all"


# --------------------------------------------------------------------------- #
# Checkpoint loading (reimplemented from src/eval/cross_dataset.py to avoid its
# broken CrossEvalRecord import -- see module docstring)
# --------------------------------------------------------------------------- #


def _checkpoint_path(
    output_root: Path,
    family: str,
    mode: str,
    train_dataset: str,
    component: str,
    profile: str,
    source_job_id: str,
) -> Path:
    """Mirrors src/eval/cross_dataset.py::_checkpoint_path."""
    return (
        output_root
        / "outputs"
        / family
        / mode
        / train_dataset
        / component
        / profile
        / source_job_id
        / "checkpoints"
        / "model.pt"
    )


@dataclass
class LoadedSurrogate:
    """A trained src/ surrogate model ready for per-region batched inference."""

    target_name: str
    input_style: str
    mode: str  # "single" or "multi"
    model: Any  # a src.models.base.SurrogateModel instance
    input_dim: int
    head_slice: tuple[int, int] | None  # None for single-head checkpoints
    checkpoint_path: Path

    @property
    def full_width(self) -> bool:
        """Whether this surrogate expects the unsliced full per-region input vector.

        True for region_full_graph targets (action/maintenance) and for *any* multi-head
        checkpoint (trained with component=None, i.e. no node_input_columns slicing, since one
        shared trunk feeds every head regardless of each head's own input_style).
        """
        return self.mode == "multi" or self.input_style == "region_full_graph"

    @property
    def is_multi_head(self) -> bool:
        return self.head_slice is not None

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference for a batch of rows. `x` is shape (batch, input_dim), raw/unstandardized.

        Returns a (batch, target_width) tensor in real units (already inverse-transformed; sliced
        to this target's head if this is a multi-head checkpoint).
        """
        standardizer = getattr(self.model, "standardizer", None)
        network = getattr(self.model, "network", None)
        if standardizer is not None and network is not None:
            # Fast path (MLP/Auto/VAE/GNN/MultiHead): direct tensor forward, no DataLoader.
            network.eval()
            with torch.no_grad():
                x_std = standardizer.transform_inputs(x.to(dtype=torch.float32))
                y_std = network(x_std)
                y = standardizer.inverse_targets(y_std)
        else:
            # Fallback for families without a plain nn.Module (e.g. GP/XGB): go through the
            # model's own batched predict() API with a throwaway-target loader.
            from torch.utils.data import DataLoader, TensorDataset

            dummy_target_width = self.model.output_dim
            loader = DataLoader(
                TensorDataset(
                    x.to(dtype=torch.float32), torch.zeros(x.shape[0], dummy_target_width)
                ),
                batch_size=x.shape[0],
            )
            y_pred, _ = self.model.predict(loader)
            y = y_pred

        if self.head_slice is not None:
            start, stop = self.head_slice
            y = y[:, start:stop]
        return y


def _entry_is_configured(entry: Mapping[str, Any] | None) -> bool:
    if not entry:
        return False
    job_id = str(entry.get("job_id", "")).strip()
    return bool(job_id) and job_id.upper() != "REPLACE_ME"


def load_surrogate_component(
    entry: Mapping[str, Any] | None,
    *,
    target_name: str,
    output_root: Path,
) -> Optional[LoadedSurrogate]:
    """Resolve one `env.surrogate.<target_name>` config entry into a loaded model, or None.

    `entry` fields:
        job_id (str): SOURCE_JOB_ID of the trained run (required; "REPLACE_ME"/empty -> None).
        family (str): MLP/Auto/VAE/GP/XGB/GNN.
        mode (str): "single" or "multi".
        train_dataset (str): dataset name the checkpoint was trained on.
        profile (str): model profile name.
        component (str, optional): overrides `target_name` for locating a single-head
            checkpoint filed under a different component name. Rarely needed.
        head (str, optional): for mode="multi", which head within the multi-head checkpoint
            corresponds to `target_name`. Defaults to `target_name`.
    """
    if not _entry_is_configured(entry):
        return None

    from src.config.schema import ModelTrainingConfig
    from src.config.load import FAMILY_PROFILE_TYPES, canonical_family
    from src.models.registry import create_multihead_surrogate_model, create_surrogate_model

    job_id = str(entry["job_id"]).strip()
    family = canonical_family(str(entry["family"]))
    mode = str(entry.get("mode", "single")).strip().lower()
    train_dataset = str(entry["train_dataset"]).strip()
    profile_name = str(entry["profile"]).strip()

    if mode not in ("single", "multi"):
        raise ValueError(f"surrogate[{target_name!r}].mode must be 'single' or 'multi', got {mode!r}")

    component = str(entry.get("component", target_name)).strip().lower()
    checkpoint_component = MULTI_COMPONENT_LABEL if mode == "multi" else component

    path = _checkpoint_path(
        output_root=output_root,
        family=family,
        mode=mode,
        train_dataset=train_dataset,
        component=checkpoint_component,
        profile=profile_name,
        source_job_id=job_id,
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Surrogate checkpoint not found for target={target_name!r}: {path} "
            f"(family={family!r}, mode={mode!r}, train_dataset={train_dataset!r}, "
            f"component={checkpoint_component!r}, profile={profile_name!r}, job_id={job_id!r}). "
            "Check results/runs/*.yaml for the correct fields."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    profile_payload = checkpoint.get("profile")
    if not isinstance(profile_payload, dict):
        raise ValueError(f"Checkpoint at {path} does not carry a model profile.")
    profile_type = FAMILY_PROFILE_TYPES[family]
    model_config = ModelTrainingConfig(
        family=family,
        profile=profile_name,
        parameters=profile_type.model_validate(profile_payload),
    )

    if mode == "multi":
        head_output_dims = {str(k): int(v) for k, v in checkpoint["head_output_dims"].items()}
        head_slices = {
            str(k): (int(v[0]), int(v[1])) for k, v in checkpoint["head_slices"].items()
        }
        head_name = str(entry.get("head", target_name))
        if head_name not in head_slices:
            raise KeyError(
                f"surrogate[{target_name!r}] mode='multi' checkpoint {path} has heads "
                f"{sorted(head_slices)}, but head={head_name!r} was requested. Set "
                f"surrogate.{target_name}.head to one of the available heads."
            )
        model = create_multihead_surrogate_model(
            config=model_config,
            input_dim=input_dim,
            head_output_dims=head_output_dims,
            head_slices=head_slices,
            graph_search_roots=None,
        )
        head_slice = head_slices[head_name]
    else:
        output_dim = int(checkpoint["output_dim"])
        model = create_surrogate_model(
            config=model_config,
            input_dim=input_dim,
            output_dim=output_dim,
            graph_search_roots=None,
        )
        head_slice = None

    model.load(path)
    logger.info(
        "[surrogate_runtime] loaded target=%s family=%s mode=%s job_id=%s checkpoint=%s",
        target_name, family, mode, job_id, path,
    )
    return LoadedSurrogate(
        target_name=target_name,
        input_style=TARGET_INPUT_STYLE[target_name],
        mode=mode,
        model=model,
        input_dim=input_dim,
        head_slice=head_slice,
        checkpoint_path=path,
    )


def load_all_surrogates(
    surrogate_cfg: Mapping[str, Any] | None,
    *,
    output_root: Path,
) -> dict[str, LoadedSurrogate]:
    """Load every configured entry under `env.surrogate`. Missing/placeholder entries are
    skipped (that target keeps using the real simulation)."""
    loaded: dict[str, LoadedSurrogate] = {}
    if not surrogate_cfg:
        return loaded
    for target_name in ALL_TARGETS:
        entry = surrogate_cfg.get(target_name)
        result = load_surrogate_component(entry, target_name=target_name, output_root=output_root)
        if result is not None:
            loaded[target_name] = result
    return loaded


# --------------------------------------------------------------------------- #
# Observation -> surrogate input adapter
# --------------------------------------------------------------------------- #


def _build_action_onehot(env: Any, num_regions: int) -> np.ndarray:
    """One-hot (num_regions, action_classes) of the action index most recently applied to each
    region, per `src/data/contract.py`'s trailing `action_onehot` x-feature. Populated by the
    always-installed step_process_actions stash (see `install_action_and_stash_patch`); before
    the first action has been recorded for this episode, falls back to an all-zero row per
    region (best-effort -- "no action" is not itself one of the action classes in MAAT's
    encoding, so this is an approximation for that edge case only).
    """
    action_classes = len(env.action_set.types)
    actions = getattr(env, "_surrogate_last_actions", None)
    if actions is None:
        return np.zeros((num_regions, action_classes), dtype=np.float32)
    actions_array = np.asarray(actions, dtype=np.int64).reshape(-1)
    return np.eye(action_classes, dtype=np.float32)[actions_array]


def build_surrogate_input(env: Any, *, full_width: bool) -> torch.Tensor:
    """Builds a (num_regions, D) batch: one row per region, matching the per-region training
    sample layout used by `src/data/assembly.py::observation_to_xy`/`feature_to_nodes`.

    D = 1[rain] + num_modes[zone_water_depth] + num_modes*(num_actions-1)[zone_modifiers]
        for mode_depth/mode_infra single-head targets (year and action one-hot dropped, per
        `src/data/contract.py::node_input_columns`), or
      = 1[year] + 1[rain] + num_modes[zone_water_depth] + num_modes*(num_actions-1)[zone_modifiers]
        + action_classes[action one-hot]
        when `full_width=True` (region_full_graph targets, and any multi-head checkpoint).

    Only [year, rain, zone_water_depth] and [zone_modifiers] come from `env._get_obs()`; any
    reward-component impact columns in between (fed back into the RL observation from the
    *previous* step) are skipped, exactly as legacy `step_transport_surrogate_model.
    _prepare_input_data()` already does via its own part1/part2 slice.
    """
    num_regions = len(env.taz_zones_ids)
    num_modes = len(env.transport_modes)
    num_action_modifiers = len(env.action_set.types) - 1  # zone_modifiers excludes the no-op action

    obs = np.asarray(env._get_obs(), dtype=np.float32).reshape(-1)

    year = obs[0]
    rain = obs[1]
    water_depth_size = num_regions * num_modes
    zone_water_depth_flat = obs[2 : 2 + water_depth_size]

    modifiers_size = num_regions * num_modes * num_action_modifiers
    zone_modifiers_flat = obs[-modifiers_size:]

    # Raw layout is mode-major (flat index = mode*num_regions + region); transpose to
    # region-major (one row per region), matching feature_to_nodes()'s spec.permute=[1, 0].
    zone_water_depth = zone_water_depth_flat.reshape(num_modes, num_regions).T  # (regions, modes)
    zone_modifiers = (
        zone_modifiers_flat.reshape(num_modes, num_regions, num_action_modifiers)
        .transpose(1, 0, 2)
        .reshape(num_regions, num_modes * num_action_modifiers)
    )
    rain_col = np.full((num_regions, 1), rain, dtype=np.float32)

    if full_width:
        year_col = np.full((num_regions, 1), year, dtype=np.float32)
        action_onehot = _build_action_onehot(env, num_regions)
        parts = [year_col, rain_col, zone_water_depth, zone_modifiers, action_onehot]
    else:
        parts = [rain_col, zone_water_depth, zone_modifiers]

    x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    return torch.as_tensor(x)


def _mode_region_from_batch(y: torch.Tensor) -> np.ndarray:
    """(num_regions, num_modes) real-valued prediction -> (num_modes, num_regions), the per-mode
    layout `compute_impacts_from_predictions`/`predictions_formatted` expect."""
    return y.detach().cpu().numpy().T


# --------------------------------------------------------------------------- #
# Monkey-patches
# --------------------------------------------------------------------------- #


def econ_triad_configured(loaded: Mapping[str, LoadedSurrogate]) -> bool:
    return all(target in loaded for target in ECON_TARGETS)


def install_econ_transport_patch(loaded: Mapping[str, LoadedSurrogate]) -> None:
    """Class-level patch: replaces `BasicEnvironment.initialize_gp_surrogate` and
    `.step_transport_surrogate_model` so that `use_surrogate=True` on the constructor runs the
    delay/cancel/infra src/ surrogates instead of the (missing, legacy) ICSCollectedGP path.

    Must be called before any `BasicEnvironment` is constructed. Only call this when
    `econ_triad_configured(loaded)` is True -- MAAT's transport step can't be partially
    substituted, so delay/cancel/infra are all-or-nothing.
    """
    import maat.world as world

    if not econ_triad_configured(loaded):
        raise ValueError("install_econ_transport_patch requires delay, cancel, and infra all loaded")

    def _patched_initialize_gp_surrogate(self):
        # Real init imports the (absent in this checkout) external_packages.ics_surrogates
        # legacy GP package. We don't need any of that: the loaded src/ models are already
        # captured in the `loaded` closure above.
        logger.info("[surrogate_runtime] initialize_gp_surrogate patched (econ triad active)")

    def _patched_step_transport_surrogate_model(self):
        # Keep the transport model's internal water-depth state consistent with the real
        # simulator path, in case anything downstream (rendering, other components) reads it.
        new_water_depths = self.retrieve_new_water_depths()
        self.model_transp.set_water_depth(new_water_depths)

        predictions = {}
        for target_name in ECON_TARGETS:
            surrogate = loaded[target_name]
            x = build_surrogate_input(self, full_width=surrogate.full_width)
            y = surrogate.predict(x)
            predictions[target_name] = _mode_region_from_batch(y)  # (num_modes, num_regions)

        predictions_formatted = {}
        for mode_index, transport_mode in enumerate(self.transport_modes):
            predictions_formatted[str(transport_mode)] = {
                "indirect_damage_delays": predictions["delay"][mode_index],
                "indirect_damage_notravel": predictions["cancel"][mode_index],
                "direct_damage_infrastructure": predictions["infra"][mode_index],
            }

        self.model_impacts.model_impacts_transport.compute_impacts_from_predictions(
            predictions_formatted, self.exposures
        )

    world.BasicEnvironment.initialize_gp_surrogate = _patched_initialize_gp_surrogate
    world.BasicEnvironment.step_transport_surrogate_model = _patched_step_transport_surrogate_model
    logger.info("[surrogate_runtime] econ transport patch installed (delay, cancel, infra)")


def install_qol_patch(loaded: Mapping[str, LoadedSurrogate]) -> None:
    """Class-level patch: replaces `QualityOfLifeModel.wb_step` so that, when a `qol` surrogate
    is loaded, it sets `self.quality_of_life_index` directly instead of running the real
    per-capita/accessibility pipeline. `ImpactsWellbeing.compute_quality_of_life_impacts()`
    (maat/impacts.py) just reads that attribute afterwards, so nothing downstream changes.

    No-ops (delegates to the original method) if `qol` isn't configured.
    """
    import maat.quality_of_life as qol_module

    if "qol" not in loaded:
        logger.info("[surrogate_runtime] qol not configured; wb_step left untouched")
        return

    original_wb_step = qol_module.QualityOfLifeModel.wb_step
    surrogate = loaded["qol"]

    def _patched_wb_step(self):
        env = self.world
        if not env.WellbeingQualityOfLifeIsReward:
            return original_wb_step(self)

        x = build_surrogate_input(env, full_width=surrogate.full_width)
        y = surrogate.predict(x)
        per_mode = _mode_region_from_batch(y)  # (num_modes, num_regions)
        # Aggregate across modes to a single per-zone index, mirroring
        # aggregate_quality_of_life_index()'s output shape (n_taz,).
        self.quality_of_life_index = np.nansum(per_mode, axis=0)

        # FGT rewards (if also active) still need the real accessibility computation; the qol
        # surrogate only covers WellbeingQualityOfLife.
        if env.WellbeingFGTCumulativeIsReward or env.WellbeingFGTGravityIsReward:
            if env.WellbeingFGTCumulativeIsReward:
                self.compute_fgt_accessibility(weighted=True, accessibility_type="FGT_cumulative")
            if env.WellbeingFGTGravityIsReward:
                self.compute_fgt_accessibility(weighted=True, accessibility_type="FGT_gravity")
            self.aggregate_fgt()
            self.aggregate_fgt_sum_over_pois()

    qol_module.QualityOfLifeModel.wb_step = _patched_wb_step
    logger.info("[surrogate_runtime] qol patch installed")


def install_action_and_stash_patch(loaded: Mapping[str, LoadedSurrogate]) -> None:
    """Always-installed class-level patch wrapping `BasicEnvironment.step_process_actions`.

    Unconditionally stashes the per-region action array on the instance (`_surrogate_last_actions`)
    so `build_surrogate_input` can build the action one-hot for full-width inputs used later in
    the same `step()` call (transport/qol hooks run after this). Additionally, for whichever of
    `action`/`maintenance` is configured, replaces the analytic per-action cost sum with the
    surrogate's prediction -- expected to stay unused in practice (action/maintenance costs are
    already O(1) lookups, not simulation-dependent), but supported for symmetry with the `src/`
    component list and multi-head QoL checkpoints that predict them alongside `qol`.
    """
    import maat.world as world

    original_step_process_actions = world.BasicEnvironment.step_process_actions
    override_targets = [name for name in ("action", "maintenance") if name in loaded]

    def _patched_step_process_actions(self, actions_to_apply):
        self._surrogate_last_actions = actions_to_apply
        result = original_step_process_actions(self, actions_to_apply)

        if override_targets:
            num_regions = len(self.taz_zones_ids)
            for target_name in override_targets:
                surrogate = loaded[target_name]
                x = build_surrogate_input(self, full_width=surrogate.full_width)
                y = surrogate.predict(x)
                prediction = y.detach().cpu().numpy().reshape(num_regions)
                impacts_key = TARGET_TO_IMPACTS_KEY[target_name]
                self.impacts[impacts_key][self.period, :] = prediction

        return result

    world.BasicEnvironment.step_process_actions = _patched_step_process_actions
    if override_targets:
        logger.info(
            "[surrogate_runtime] action/maintenance patch installed for targets=%s", override_targets
        )
    else:
        logger.info(
            "[surrogate_runtime] step_process_actions patched to stash actions only "
            "(no action/maintenance override configured)"
        )


def install_all_patches(loaded: Mapping[str, LoadedSurrogate]) -> bool:
    """Installs whichever patches are warranted by `loaded`. Returns True if the econ transport
    triad is active (i.e. the caller should also pass `use_surrogate=True` to
    `BasicEnvironment`), False otherwise.

    Safe to call with an empty `loaded` dict: every patch below no-ops (or, for the always-on
    action-stash wrapper, has no observable effect) in that case, so behaviour is identical to
    the unpatched `maat` package. Must run before any `BasicEnvironment` is constructed.
    """
    econ_active = econ_triad_configured(loaded)
    if econ_active:
        install_econ_transport_patch(loaded)
    else:
        missing = [target for target in ECON_TARGETS if target not in loaded]
        if any(target in loaded for target in ECON_TARGETS):
            logger.warning(
                "[surrogate_runtime] econ triad partially configured (missing=%s); "
                "falling back to the real transport simulation for delay/cancel/infra",
                missing,
            )

    install_qol_patch(loaded)
    install_action_and_stash_patch(loaded)
    return econ_active
