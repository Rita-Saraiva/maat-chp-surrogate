"""
Phase 2: wall-clock comparison between the real MAAT simulation and the surrogate-backed path
installed by src/surrogate_runtime.py, on the same RL config and action sequence.

Because surrogate_runtime's patches are installed at the *class* level (they replace methods on
maat.world.BasicEnvironment / maat.quality_of_life.QualityOfLifeModel for the whole process), the
"simulation" and "surrogate" conditions must not share a process -- otherwise the second
condition run would inherit whatever the first one patched. This script runs each condition in
its own subprocess (a fresh, unpatched import of `maat` every time) and then combines the two
raw timing files into one CSV + printed summary in the parent process.

`overall_script_modified.py`/`overall_script_surrogate.py` (the two "driver" scripts this reuses
`make_env()` from) live wherever your RL job's own working directory is -- NOT under this repo's
`src/`, and (per the same caveat as overall_script_surrogate.pbs) `RL_framework/` in this repo
checkout is a local reference copy only, not a real cluster path. Point `--driver-dir` at wherever
you've actually deployed those two files (defaults to this repo's own `RL_framework/`, which only
works for local/dev checkouts laid out the same way).

Usage (inside the MAAT container -- see benchmark_surrogate_vs_simulation.pbs). Note /mnt/project
is module-provided on this cluster (bound to maat_rl_intro via maat-container-loader, where
`import maat` resolves from) -- this repo's own src/, configs/overall_surrogate/, and outputs/
are bound separately, to /mnt/surrogate; see overall_script_surrogate.pbs for the full
explanation:

    python src/benchmark_surrogate_vs_simulation.py \
        --driver-dir /mnt/work \
        --config-path /mnt/surrogate/configs/overall_surrogate \
        --config-name economic_surrogate_cph_rcp26 \
        --steps 50 --warmup 5 \
        --out results/tables/surrogate_timing

Writes:
    <out>.csv            one row per (condition, step_index, seconds)
    <out>_summary.json    mean/median/p95/total per condition + speedup factor

Single-condition runs (e.g. to inspect one side in isolation) are also supported directly via
--mode simulation|surrogate --out-raw <path>, which is what the parent process invokes internally.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DRIVER_DIR = REPO_ROOT / "RL_framework"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--driver-dir",
        default=str(DEFAULT_DRIVER_DIR),
        help=(
            "Directory containing overall_script_modified.py, overall_script_surrogate.py, and "
            "a `maat/` package importable from it. Defaults to this repo's RL_framework/ (local/dev "
            "checkouts only) -- on the cluster, point this at your job's actual working directory "
            "(e.g. /mnt/work)."
        ),
    )
    parser.add_argument("--config-path", required=True, help="Directory containing the Hydra env config, e.g. configs")
    parser.add_argument("--config-name", required=True, help="Config file name without .yaml, e.g. economic_surrogate_cph_rcp26")
    parser.add_argument("--steps", type=int, default=30, help="Number of *timed* env.step() calls per condition")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed env.step() calls before timing starts")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the shared random action sequence")
    parser.add_argument("--out", default="results/tables/surrogate_timing", help="Output path prefix (no extension) for the combined CSV/summary; used only when --mode is not set")
    parser.add_argument(
        "--mode",
        choices=["simulation", "surrogate"],
        default=None,
        help="Run a single condition directly instead of orchestrating both (internal use by the parent process; also usable standalone).",
    )
    parser.add_argument("--out-raw", default=None, help="Where to write this condition's raw per-step timings as JSON (required with --mode)")
    return parser.parse_args()


def _load_env_and_actions(mode: str, driver_dir: Path, config_path: Path, config_name: str, seed: int):
    """Builds one Monitor-wrapped BasicEnvironment for `mode` and a matching action sequence.

    Imports overall_script_modified.py (real simulation, never touches surrogate_runtime) or
    overall_script_surrogate.py (installs whichever src.surrogate_runtime patches the config's
    env.surrogate block configures) from `driver_dir`, depending on `mode`, and reuses that
    module's own make_env() -- the exact same env-construction path real training runs, so the
    benchmark reflects real per-step cost.
    """
    sys.path.insert(0, str(driver_dir))
    from omegaconf import OmegaConf

    if mode == "simulation":
        import overall_script_modified as driver  # noqa: PLC0415
    elif mode == "surrogate":
        import overall_script_surrogate as driver  # noqa: PLC0415
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    cfg_file = Path(config_path) / f"{config_name}.yaml"
    cfg = OmegaConf.load(cfg_file)

    rank = int(cfg.training.env_starting_index)
    env = driver.make_env(rank, cfg)()  # _init() already calls env.reset()

    import numpy as np

    rng = np.random.default_rng(seed)
    action_space = env.action_space
    return env, action_space, rng


def run_condition(
    mode: str, driver_dir: str, config_path: str, config_name: str, steps: int, warmup: int, seed: int
) -> list[float]:
    env, action_space, rng = _load_env_and_actions(mode, Path(driver_dir), Path(config_path), config_name, seed)

    action_sequence = [action_space.sample() for _ in range(warmup + steps)]

    for i in range(warmup):
        obs, reward, done, truncated, info = env.step(action_sequence[i])
        if done or truncated:
            env.reset()

    per_step_seconds: list[float] = []
    for i in range(warmup, warmup + steps):
        start = time.perf_counter()
        obs, reward, done, truncated, info = env.step(action_sequence[i])
        per_step_seconds.append(time.perf_counter() - start)
        if done or truncated:
            env.reset()

    return per_step_seconds


def _summarize(label: str, seconds: list[float]) -> dict:
    sorted_seconds = sorted(seconds)
    p95_index = max(0, int(round(0.95 * (len(sorted_seconds) - 1))))
    return {
        "condition": label,
        "n_steps": len(seconds),
        "mean_seconds": statistics.mean(seconds),
        "median_seconds": statistics.median(seconds),
        "p95_seconds": sorted_seconds[p95_index],
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "total_seconds": sum(seconds),
    }


def _write_csv(out_csv: Path, per_condition: dict[str, list[float]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "step_index", "seconds"])
        for condition, seconds_list in per_condition.items():
            for index, seconds in enumerate(seconds_list):
                writer.writerow([condition, index, f"{seconds:.6f}"])


def _orchestrate(args: argparse.Namespace) -> None:
    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    per_condition: dict[str, list[float]] = {}
    summaries: dict[str, dict] = {}

    for mode in ("simulation", "surrogate"):
        raw_path = out_prefix.parent / f"{out_prefix.name}_{mode}_raw.json"
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--driver-dir", args.driver_dir,
            "--config-path", args.config_path,
            "--config-name", args.config_name,
            "--steps", str(args.steps),
            "--warmup", str(args.warmup),
            "--seed", str(args.seed),
            "--mode", mode,
            "--out-raw", str(raw_path),
        ]
        print(f"[benchmark] running condition={mode!r}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        per_condition[mode] = json.loads(raw_path.read_text(encoding="utf-8"))
        summaries[mode] = _summarize(mode, per_condition[mode])

    _write_csv(Path(f"{args.out}.csv"), per_condition)

    speedup = None
    if summaries["surrogate"]["mean_seconds"] > 0:
        speedup = summaries["simulation"]["mean_seconds"] / summaries["surrogate"]["mean_seconds"]
    summary_payload = {"conditions": summaries, "speedup_simulation_over_surrogate": speedup}
    Path(f"{args.out}_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SURROGATE VS SIMULATION TIMING")
    print("=" * 72)
    for mode in ("simulation", "surrogate"):
        summary = summaries[mode]
        print(
            f"{mode:>10}: mean={summary['mean_seconds']*1000:8.2f}ms  "
            f"median={summary['median_seconds']*1000:8.2f}ms  "
            f"p95={summary['p95_seconds']*1000:8.2f}ms  "
            f"total={summary['total_seconds']:8.2f}s over {summary['n_steps']} steps"
        )
    if speedup is not None:
        print(f"\nspeedup (simulation mean / surrogate mean): {speedup:.2f}x")
    print(f"\nCSV:     {args.out}.csv")
    print(f"Summary: {args.out}_summary.json")
    print("=" * 72)


def main() -> None:
    args = _parse_args()
    if args.mode is not None:
        if not args.out_raw:
            raise SystemExit("--out-raw is required when --mode is set")
        seconds = run_condition(
            args.mode, args.driver_dir, args.config_path, args.config_name, args.steps, args.warmup, args.seed
        )
        out_raw = Path(args.out_raw)
        out_raw.parent.mkdir(parents=True, exist_ok=True)
        out_raw.write_text(json.dumps(seconds), encoding="utf-8")
        return
    _orchestrate(args)


if __name__ == "__main__":
    main()
