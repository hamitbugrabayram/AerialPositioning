#!/usr/bin/env bash
# Sequential evaluation for all regions

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Conda initialization (portable)
if ! command -v conda >/dev/null 2>&1; then
  if [[ -n "${CONDA_EXE:-}" ]]; then
    # shellcheck disable=SC1090
    source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda initialization script not found."
    echo "Install conda or set CONDA_EXE before running this script."
    exit 1
  fi
fi

ENV_NAME="${CONDA_ENV_NAME:-satellite-loc}"
conda activate "$ENV_NAME"

echo "=========================================="
echo "Starting dataset preparation at $(date)"
echo "=========================================="
python runner.py --dataset-prepare all --tile-provider google esri --map-margin 3000
echo "Dataset preparation DONE at $(date)"

for region in $(seq 1 11); do
  echo "=========================================="
  echo "Starting Region ${region} (auto zoom) at $(date)"
  echo "=========================================="
  python runner.py --dataset-eval "$region" --tile-provider google esri
  echo "Region ${region} DONE at $(date)"
done

echo "=========================================="
echo "ALL REGIONS COMPLETE at $(date)"
echo "=========================================="
