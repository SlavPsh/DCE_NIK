#!/bin/bash
#SBATCH --job-name=nik-xcat
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:30
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/rnga/vvpshenov/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
MODEL="$1"; TAG="$2"; shift 2
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /scratch/rnga/vvpshenov/DCE_NIK
micromamba run -n torch29 python xcat_train.py --model "$MODEL" --save-dir results_xcat_${TAG} "$@"
echo "EXIT $?"
