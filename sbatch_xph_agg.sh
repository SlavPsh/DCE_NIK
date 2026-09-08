#!/bin/bash
#SBATCH --job-name=xph-agg
#SBATCH --partition=luna-cpu-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/agg_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[agg $SLURM_JOB_ID] $(date)"
$MM run -n torch29 python -u xph_aggregate.py
echo "[agg $SLURM_JOB_ID] DONE $(date)"
