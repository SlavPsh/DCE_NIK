#!/bin/bash
#SBATCH --job-name=t4-gpurecon
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/task4_xcat_nomotion_pilot/logs/sbatch_gpu_recon_%j.log
# Task 4 non-neural recons on a full GPU (direct-discrete + CS-then-fit + NUFFT, both fracs).
set -e
cd /scratch/rnga/vvpshenov/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[job $SLURM_JOB_ID] host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date)"
$MM run -n torch29 python -u task4_gpu_recon.py --frac both
echo "[job $SLURM_JOB_ID] DONE $(date)"
