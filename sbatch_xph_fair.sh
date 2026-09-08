#!/bin/bash
#SBATCH --job-name=xph-fair
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/fair_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[fair $SLURM_JOB_ID] gpu=$CUDA_VISIBLE_DEVICES $(date)"
$MM run -n torch29 python -u xph_grasp_fair.py
echo "[fair $SLURM_JOB_ID] DONE $(date)"
