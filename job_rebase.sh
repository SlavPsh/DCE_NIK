#!/bin/bash
#SBATCH --job-name=nik-rebase
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-02:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/slurm-%x-%j.out
#SBATCH --error=/scratch/rnga/vvpshenov/DCE_NIK/slurm-%x-%j.out
set -uo pipefail
TAG="$1"; shift
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /scratch/rnga/vvpshenov/DCE_NIK
# re-baseline at support_radius=1.0 (render fix now default).
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 13 --save-dir /scratch/rnga/vvpshenov/DCE_NIK/results_rebase_${TAG} \
  --resume --no-compile "$@"
echo "EXIT $?"
