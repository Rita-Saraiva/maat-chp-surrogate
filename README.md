# Surrogate Models for RL Climate-Adaptation Environments

Surrogate (fast approximate) models that predict the outcomes of a reinforcement-learning
climate-adaptation simulator over a 278-region graph of the Greater Copenhagen area. Given
an environment state and action, a trained surrogate predicts the objective outputs
(e.g. economic *delay* / *cancel* / *infra*, or quality-of-life *qol* / *action* /
*maintenance*) without running the full simulation.

The framework builds tabular datasets from cached RL environment buffers, trains one of
several model families, and evaluates them (including cross-dataset generalisation,
datasize-robustness sweeps, and prediction/dataset analysis plots).

## Model families

| `MODEL` | Family | Extra dependency |
|---------|--------|------------------|
| `MLP`   | Multi-layer perceptron | — |
| `Auto`  | Bottleneck autoencoder-style regressor | — |
| `VAE`   | Variational autoencoder regressor (KL warm-up) | — |
| `GNN`   | Graph neural network over the region graph | — |
| `GP`    | Sparse variational Gaussian process (LMC) | `gpytorch` |
| `XGB`   | Gradient-boosted trees (native multi-output) | `xgboost>=2.0` |

Each family supports **single** mode (one component) and **multi** mode (a shared trunk
with one head per component of an objective).

## Repository layout

```
configs/     YAML configuration
  data/        dataset-build definitions and raw-source roots
  graph/       region_graph.yaml (nodes + edges of the 278-region graph)
  model/       per-family model profiles (mlp/auto/vae/gp/xgb/gnn.yaml)
  train/       default training hyper-parameters
src/         Python package (imported as `src.*`)
  config/      config schema + environment-driven loaders
  data/        dataset building, sharding, loaders, graph adjacency
  models/      model families + registry
  train/       training loop, standardization, trainer entrypoint
  eval/        metrics, cross-dataset evaluation, analysis helpers
  experiments/ datasize-robustness plotting
  results/     results tables (run_history.pkl) writers
jobs/        PBS batch scripts + container dependency installer
tests/       smoke tests (one per model family / feature)
notebooks/   helper notebooks (qsub command generation, analysis)
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`xgboost`, `gpytorch`, and `scipy` are optional and only needed for the `XGB` / `GP`
families and the policy rank-correlation metrics, respectively.

## Running

Every entrypoint is driven by environment variables so the same command runs locally and
inside the cluster container. `src` must be importable (run from the repository root, or
set `PYTHONPATH` to it).

**Train a surrogate:**

```bash
OBJECTIVE=Econ DATASET=econ_combined MODE=single MODEL=MLP \
  PARAMETERS=simple_mlp COMPONENT=delay LOADER_MODE=chunked \
  python -u -m src.train_surrogate
```

Required: `OBJECTIVE`, `DATASET`, `MODE` (`single`/`multi`), `MODEL`, `PARAMETERS`
(a profile name from `configs/model/<family>.yaml`), and `COMPONENT` (single mode only).
Optional: `LOADER_MODE`, `BATCH_SIZE`, `NUM_WORKERS`, `MAX_SAMPLES`, `TRAIN_FRACTION`,
`MODEL_CONFIG`, `TRAIN_CONFIG`.

Other entrypoints (same env-var style):

- `python -u -m src.evaluate_surrogate` — score an existing checkpoint on another dataset's test split.
- `python -u -m src.analyze_dataset` — dataset statistics, histograms, and water-depth plots.
- `python -u -m src.plot_predictions` — predicted-vs-true plots for a checkpoint.
- `python -m src.experiments.plot_robustness` — datasize-robustness curves.

## Results

Aggregated metrics are written to `results/tables/run_history.pkl` (plus per-mode and
datasize-robustness tables), rebuilt race-free from per-job JSON shards under
`results/tables/`. Trained checkpoints live under `outputs/<family>/<mode>/<dataset>/...`.

## Cluster (PBS + Singularity)

The `jobs/*.pbs` scripts submit each entrypoint to a PBS queue, forwarding the environment
variables into the `maat.sif` Singularity container. Set `PROJECT_ROOT` at the top of a
script (or export it) to your code checkout, then e.g.:

```bash
OBJECTIVE=Econ DATASET=econ_combined MODE=single MODEL=MLP \
  PARAMETERS=simple_mlp COMPONENT=delay LOADER_MODE=chunked \
  qsub -v OBJECTIVE,DATASET,MODE,MODEL,PARAMETERS,COMPONENT,LOADER_MODE jobs/train_surrogate.pbs
```

`notebooks/Calls_to_run.ipynb` generates ready-to-paste `qsub` commands for batches of runs.

## Tests

Smoke tests build tiny synthetic datasets and exercise each family end-to-end:

```bash
python -m tests.test_multihead_smoke
python -m tests.test_gnn_smoke
python -m tests.test_xgboost_smoke      # requires xgboost>=2.0
```

They require `pydantic>=2`. On Windows, set `KMP_DUPLICATE_LIB_OK=TRUE` before running to
avoid the MKL/torch duplicate-OpenMP error.
