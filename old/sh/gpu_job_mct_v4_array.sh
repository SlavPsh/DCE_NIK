#!/bin/bash
#SBATCH --job-name=nik-mct-v4
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-03:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -eu
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
nvidia-smi || true

export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

cd /scratch/rnga/vvpshenov/DCE_NIK

CONFIGS=(
    config/mct_v4_h192_d8_lr3e-6.toml
    config/mct_v4_h256_d8_lr1e-5.toml
    config/mct_v4_h256_d8_lr3e-6.toml
    config/mct_v4_h320_d8_lr1e-5.toml
    config/mct_v4_h320_d8_lr3e-6.toml
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CFG"

python train_multicoil_cart.py "$CFG" --single
