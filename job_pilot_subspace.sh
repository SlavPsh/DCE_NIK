#!/bin/bash
#SBATCH --job-name=nik-subR16
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/slurm-%x-%j.out
#SBATCH --error=/scratch/rnga/vvpshenov/DCE_NIK/slurm-%x-%j.out
set -uo pipefail
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /scratch/rnga/vvpshenov/DCE_NIK
# PILOT: factorized low-rank NIK, rank 16, warm-started Phi. Slice 13, winning recipe.
# de-risks the R-sweep. Baseline (full-rank wire_ff_res): held-out 0.3245.
micromamba run -n torch29 python train_grasp_nik.py \
  --model wire_ff_subspace --rank 16 \
  --slices 13 \
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_nik_subspace_r16 \
  --resume --no-compile
echo "EXIT $?"
