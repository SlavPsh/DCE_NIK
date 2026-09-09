#!/bin/bash
#SBATCH --job-name=xph-navfig
#SBATCH --partition=defq
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/navfig_%j.log
set -e; cd /net/beegfs/users/P101440/DCE_NIK
MM=/net/beegfs/users/P101440/micromamba/bin/micromamba
echo "[navfig $SLURM_JOB_ID] $(date)"
$MM run -n torch29 python -u xph_nav_figure.py
echo "[navfig $SLURM_JOB_ID] DONE $(date)"
