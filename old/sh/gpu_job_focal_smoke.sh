#!/bin/bash
#SBATCH --job-name=nik-focal-smoke
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:20
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

python train_cart_eval.py config/ablation_focal_smoke_A.toml --single
