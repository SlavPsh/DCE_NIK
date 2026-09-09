#!/bin/bash
#SBATCH --job-name=xph-agg
#SBATCH --partition=defq
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/agg_%j.log
set -e; cd /net/beegfs/users/P101440/DCE_NIK
MM=/net/beegfs/users/P101440/micromamba/bin/micromamba
echo "[agg $SLURM_JOB_ID] $(date)"
$MM run -n torch29 python -u xph_aggregate.py
echo "[agg $SLURM_JOB_ID] DONE $(date)"
