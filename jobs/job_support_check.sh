#!/bin/bash
#SBATCH --job-name=support-chk
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2g.24gb:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:30
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
micromamba run -n torch29 python /net/beegfs/users/P101440/DCE_NIK/render_support_check.py
