#!/bin/bash
#SBATCH --job-name=nik-mc-cart
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-07:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
eval "$(/net/beegfs/users/P101440/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

cd /net/beegfs/users/P101440/DCE_NIK

CONFIG_PATH="${1:?Usage: sbatch gpu_job_multicoil_cart.sh CONFIG [single|sweep]}"
MODE="${2:-single}"

if [ "$MODE" = "single" ]; then
    python train_multicoil_cart.py "$CONFIG_PATH" --single
else
    python train_multicoil_cart.py "$CONFIG_PATH"
fi
