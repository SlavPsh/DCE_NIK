#!/bin/bash
#SBATCH --job-name=nik-mct-sweep
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-07:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Single slurm job that registers a wandb sweep from the TOML's [sweep] block
# and runs --count agents sequentially in this process.
#
# Usage:
#   sbatch gpu_job_mct_wandb_sweep.sh CONFIG_PATH COUNT
# e.g.
#   sbatch gpu_job_mct_wandb_sweep.sh config/mct_sweep.toml 16

set -eu
CFG="${1:?Usage: sbatch gpu_job_mct_wandb_sweep.sh CONFIG_PATH COUNT}"
COUNT="${2:?COUNT required}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CONFIG=$CFG  COUNT=$COUNT"
nvidia-smi || true

export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

cd /scratch/rnga/vvpshenov/DCE_NIK

python train_multicoil_cart.py "$CFG" --count "$COUNT"
