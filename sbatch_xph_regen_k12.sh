#!/bin/bash
#SBATCH --job-name=xph-regenk12
#SBATCH --partition=luna-cpu-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/regenk12_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[regenk12 $SLURM_JOB_ID] cv figure $(date)"
$MM run -n torch29 python -u xph_cv_figure.py
echo "[regenk12 $SLURM_JOB_ID] aggregate at K12"
$MM run -n torch29 python -u xph_aggregate.py | tail -8
echo "[regenk12 $SLURM_JOB_ID] DONE $(date)"
