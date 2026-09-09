#!/bin/bash
#SBATCH --job-name=cs-heldout
#SBATCH --partition=defq
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=0-01:30
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/net/beegfs/users/P101440/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
export PATH="/net/beegfs/users/P101440/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/net/beegfs/users/P101440/micromamba"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
micromamba run -n torch29 python /net/beegfs/users/P101440/DCE_NIK/cs_heldout_loss.py
