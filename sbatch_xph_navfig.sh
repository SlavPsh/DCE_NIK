#!/bin/bash
#SBATCH --job-name=xph-navfig
#SBATCH --partition=luna-cpu-short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/navfig_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[navfig $SLURM_JOB_ID] $(date)"
$MM run -n torch29 python -u xph_nav_figure.py
echo "[navfig $SLURM_JOB_ID] DONE $(date)"
