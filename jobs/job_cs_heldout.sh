#!/bin/bash
#SBATCH --job-name=cs-heldout
#SBATCH --partition=luna-cpu-short
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=0-01:30
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/rnga/vvpshenov/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
micromamba run -n torch29 python /scratch/rnga/vvpshenov/DCE_NIK/cs_heldout_loss.py
