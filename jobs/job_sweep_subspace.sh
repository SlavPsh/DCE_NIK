#!/bin/bash
#SBATCH --job-name=nik-subR
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
R="$1"
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /net/beegfs/users/P101440/DCE_NIK
# R-sweep: factorized low-rank NIK, warm-started Phi, slice 13, winning recipe.
# baseline (full-rank wire_ff_res): held-out 0.3245, swing 45.5%, nav 0.985.
micromamba run -n torch29 python train_grasp_nik.py \
  --model wire_ff_subspace --rank "$R" \
  --slices 13 \
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_nik_subspace_r${R} \
  --resume --no-compile
echo "EXIT $?"
