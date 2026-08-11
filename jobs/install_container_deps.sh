#!/bin/bash
# One-time installer for Python packages that are missing from the maat.sif
# container (currently: xgboost, required by the XGB surrogate family).
#
# Packages are installed into a bind-mounted project directory
# (/mnt/project/.python_deps) instead of the read-only container. The training
# job (scripts/train_surrogate.pbs) already puts this directory on PYTHONPATH.
#
# --no-deps is used on purpose: the container already provides numpy/scipy/torch,
# so only the pure xgboost wheel (which bundles libxgboost.so) is installed here.
# This avoids a second numpy/scipy on PYTHONPATH shadowing the container's copies
# and breaking the torch ABI. If a later import fails on a missing dependency
# (e.g. scipy), add it to EXTRA_NO_DEPS below.
#
# Run on a node with outbound network access (typically a login node):
#   bash scripts/install_container_deps.sh
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/projects/dtu_00074/scratch/ritsar_code/Surrogate}"
SIF="${SIF:-/home/projects/dtu_00074/apps/singularity_images/maat.sif}"
DEPS_DIR="${PROJECT_ROOT}/.python_deps"

# Packages installed with dependency resolution disabled (container provides deps).
NO_DEPS_PACKAGES=("xgboost>=2.0")
# Add here if an import complains about a missing transitive dependency, e.g.:
# EXTRA_NO_DEPS=("scipy")
EXTRA_NO_DEPS=()

module load tools singularity/4.3.0
mkdir -p "${DEPS_DIR}"

printf 'Installing into %s\n' "${DEPS_DIR}"
singularity exec \
  --cleanenv \
  --pwd /mnt/project \
  --bind "${PROJECT_ROOT}:/mnt/project" \
  "${SIF}" \
  python -m pip install --no-cache-dir --no-deps \
    --target /mnt/project/.python_deps \
    "${NO_DEPS_PACKAGES[@]}" "${EXTRA_NO_DEPS[@]}"

printf 'Verifying import with PYTHONPATH=/mnt/project/.python_deps\n'
export SINGULARITYENV_PYTHONPATH=/mnt/project/.python_deps
singularity exec \
  --cleanenv \
  --pwd /mnt/project \
  --bind "${PROJECT_ROOT}:/mnt/project" \
  "${SIF}" \
  python -u -c "import xgboost; print('xgboost', xgboost.__version__)"

printf 'Done. XGB training jobs can now import xgboost.\n'
