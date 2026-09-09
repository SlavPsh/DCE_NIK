#!/bin/bash
#SBATCH --job-name=nik-mct-sweep
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --partition=gpu
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

export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
eval "$(/net/beegfs/users/P101440/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

cd /net/beegfs/users/P101440/DCE_NIK

python train_multicoil_cart.py "$CFG" --count "$COUNT"
