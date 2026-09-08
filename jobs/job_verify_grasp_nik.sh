#!/bin/bash
#SBATCH --job-name=nik-verify
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-05:30
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/rnga/vvpshenov/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /scratch/rnga/vvpshenov/DCE_NIK
# winning-recipe DEFAULTS (wire_ff_res h512 d12, FF k256/t32, dcf0, env0.75, 40k, 70/30 split).
# target (autoresearch e8_ts1p5_tf32, same slice-13 data): held-out ~0.323, swing ~51%, nav-corr ~0.885.
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 13 \
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_nik_verify \
  --no-compile
echo "EXIT $?"
