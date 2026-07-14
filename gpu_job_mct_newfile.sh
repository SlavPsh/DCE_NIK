#!/bin/bash
#SBATCH --job-name=nik-mct-newfile
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-07:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -eu
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true

export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

cd /scratch/rnga/vvpshenov/DCE_NIK
python train_multicoil_cart.py config/mct_newfile_h384.toml --single
