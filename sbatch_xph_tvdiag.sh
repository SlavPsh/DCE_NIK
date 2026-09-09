#!/bin/bash
#SBATCH --job-name=xph-tvdiag
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/tvdiag_%j.log
set -e; cd /net/beegfs/users/P101440/DCE_NIK
MM=/net/beegfs/users/P101440/micromamba/bin/micromamba
echo "[tvdiag $SLURM_JOB_ID] $(date)"
$MM run -n torch29 python -u xph_nik_tvdiag.py
echo "[tvdiag $SLURM_JOB_ID] DONE $(date)"
