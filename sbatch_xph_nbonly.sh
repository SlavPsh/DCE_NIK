#!/bin/bash
#SBATCH --job-name=xph-nbonly
#SBATCH --partition=luna-cpu-short
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/nbonly_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[nbonly $SLURM_JOB_ID] build notebook $(date)"
$MM run -n torch29 python -u build_report_nb.py
echo "[nbonly $SLURM_JOB_ID] DONE $(date)"
