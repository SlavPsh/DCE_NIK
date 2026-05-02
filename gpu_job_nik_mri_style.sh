#!/bin/bash
#SBATCH --job-name=nik-mri-style
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-07:00
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

CONFIG_PATH="${1:?Usage: sbatch gpu_job_nik_mri_style.sh CONFIG_PATH [RUN_NAME]}"
RUN_NAME="${2:-nik_mri_style}"

echo "Config:   $CONFIG_PATH"
echo "Run name: $RUN_NAME"

"$PYTHON_BIN" train_nik_mri_style.py "$CONFIG_PATH" --run-name "$RUN_NAME"
