#!/bin/bash
#SBATCH --job-name=nik-mct-night
#SBATCH --gres=gpu:4g.40gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=48G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-04:30
#SBATCH --array=0-12
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
    config/mct_night_h192_a0p00.toml
    config/mct_night_h192_a0p25.toml
    config/mct_night_h192_a0p50.toml
    config/mct_night_h192_a1p00.toml
    config/mct_night_h256_a0p00.toml
    config/mct_night_h256_a0p25.toml
    config/mct_night_h256_a0p50.toml
    config/mct_night_h256_a1p00.toml
    config/mct_night_h320_a0p00.toml
    config/mct_night_h320_a0p25.toml
    config/mct_night_h320_a0p50.toml
    config/mct_night_h320_a1p00.toml
    config/mct_night_h256_a0p50_long200k.toml
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CFG"

python train_multicoil_cart.py "$CFG" --single
