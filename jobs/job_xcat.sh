#!/bin/bash
#SBATCH --job-name=nik-xcat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:30
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
MODEL="$1"; TAG="$2"; shift 2
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /net/beegfs/users/P101440/DCE_NIK
micromamba run -n torch29 python xcat_train.py --model "$MODEL" --save-dir results_xcat_${TAG} "$@"
echo "EXIT $?"
