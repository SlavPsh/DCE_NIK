#!/bin/bash
#SBATCH --job-name=step2regen
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/step2regen_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[step2regen $SLURM_JOB_ID] gpu=$CUDA_VISIBLE_DEVICES $(date)"
$MM run -n torch29 python -u step2_regen.py
echo "[step2regen $SLURM_JOB_ID] DONE $(date)"
