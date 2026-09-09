#!/bin/bash
#SBATCH --job-name=nik-pktv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-03:30
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
LABEL="$1"; shift                                  # remaining args -> python
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /net/beegfs/users/P101440/DCE_NIK
# P0/P3 adjudication batch: temporal TV + gamma-variate PK vs the bandlimit frontier.
# all slice 13, rank 16, winning recipe; --resume survives a wall-clock timeout.
micromamba run -n torch29 python train_grasp_nik.py \
  --slices 13 --rank 16 \
  --save-dir /net/beegfs/users/P101440/DCE_NIK/results_pktv_${LABEL} \
  --resume --no-compile "$@"
echo "EXIT $?"
