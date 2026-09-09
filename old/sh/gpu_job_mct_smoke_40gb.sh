#!/bin/bash
#SBATCH --job-name=nik-mct-smoke40
#SBATCH --gres=gpu:4g.47gb:1
#SBATCH --partition=gpu
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:30
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
python train_multicoil_cart.py config/smoke_mct.toml --single
