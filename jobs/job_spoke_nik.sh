#!/bin/bash
#SBATCH --job-name=nik-spoke
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:30
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/rnga/vvpshenov/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
LABEL="$1"; SLICES="${2:-13}"                       # LABEL = f100/f70/f50/f35/f25
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /scratch/rnga/vvpshenov/DCE_NIK
# spoke-reduction frontier: factorized R16 baseline, IDENTICAL acquired spokes as CS
# (same keep-file), trained on ALL of them (no heldout -> same input data as CS).
micromamba run -n torch29 python train_grasp_nik.py \
  --model wire_ff_subspace --rank 16 --slices "$SLICES" \
  --spoke-keep-file /scratch/rnga/vvpshenov/DCE_NIK/spoke_masks/keep_${LABEL}.npy \
  --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_spoke_nik_${LABEL} \
  --resume --no-compile
echo "EXIT $?"
