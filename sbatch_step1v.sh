#!/bin/bash
#SBATCH --job-name=step1v
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/step1v_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[step1v $SLURM_JOB_ID] gpu=$CUDA_VISIBLE_DEVICES $(date)"
$MM run -n torch29 python -u step1_verify.py
echo "[step1v $SLURM_JOB_ID] DONE $(date)"
