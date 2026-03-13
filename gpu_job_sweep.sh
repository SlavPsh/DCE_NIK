#!/bin/bash
#SBATCH --job-name=nik-sweep
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-07:00
#SBATCH --nice=10000
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# ---- Micromamba setup ----
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

which python
python --version

# ---- wandb offline mode  ----
# export WANDB_MODE=offline

cd /scratch/rnga/vvpshenov/DCE_NIK

# ---- Parse arguments ----
# Usage:
#   sbatch gpu_job_sweep.sh config/training.toml                        # creates new sweep
#   sbatch gpu_job_sweep.sh config/training.toml SWEEP_ID [COUNT]       # joins existing sweep
CONFIG_PATH="${1:?Usage: sbatch gpu_job_sweep.sh CONFIG_PATH [SWEEP_ID] [COUNT]}"
SWEEP_ID="${2:-}"
COUNT="${3:-50}"

echo "Config:    $CONFIG_PATH"
echo "Sweep ID:  ${SWEEP_ID:-<new sweep>}"
echo "Run count: $COUNT"

if [ -n "$SWEEP_ID" ]; then
    python train_wandb.py "$CONFIG_PATH" --sweep-id "$SWEEP_ID" --count "$COUNT"
else
    python train_wandb.py "$CONFIG_PATH" --count "$COUNT"
fi
