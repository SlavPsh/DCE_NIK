#!/bin/bash
#SBATCH --job-name=nikmri-smoke
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:30
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

export MPLCONFIGDIR="/home/rnga/vvpshenov/tmp/mpl"
mkdir -p "$MPLCONFIGDIR"

cd /scratch/rnga/vvpshenov/DCE_NIK

CONFIG_PATH="${1:?Usage: sbatch gpu_job_nikmri_smoke.sh CONFIG_PATH [STEPS]}"
STEPS="${2:-3}"

python train_nik_mri_style.py "$CONFIG_PATH" --single --steps "$STEPS"
