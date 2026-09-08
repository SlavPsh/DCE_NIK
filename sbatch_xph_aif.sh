#!/bin/bash
#SBATCH --job-name=xph-aif
#SBATCH --partition=luna-cpu-short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/aif_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[aif $SLURM_JOB_ID] $(date)"
$MM run -n torch29 python -u aif_build.py
echo "[aif $SLURM_JOB_ID] DONE $(date)"
