#!/bin/bash
#SBATCH --job-name=nik-mri-style-test
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:30
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
PYTHON_BIN="/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python"

export MPLCONFIGDIR="/home/rnga/vvpshenov/tmp/mpl"
mkdir -p "$MPLCONFIGDIR"

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

cd /scratch/rnga/vvpshenov/DCE_NIK

CONFIG_PATH="${1:-config/nik_mri_style_dce_test.toml}"
RUN_NAME="${2:-nik_mri_style_test}"

echo "Config:   $CONFIG_PATH"
echo "Run name: $RUN_NAME"

"$PYTHON_BIN" train_nik_mri_style.py "$CONFIG_PATH" --run-name "$RUN_NAME"
