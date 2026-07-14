#!/bin/bash
#SBATCH --job-name=nik-focal-ablation
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --partition=luna-gpu-short
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --time=0-07:00
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
    config/ablation_focal_A_baseline.toml
    config/ablation_focal_B_envelope_only.toml
    config/ablation_focal_C_full_stack.toml
    config/ablation_focal_D_no_dcf.toml
    config/ablation_focal_E_pure_focal.toml
)

CFG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
echo "Running config: $CFG"

python train_cart_eval.py "$CFG" --single
