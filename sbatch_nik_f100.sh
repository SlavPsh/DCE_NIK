#!/bin/bash
#SBATCH --job-name=t4-nikf100
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/task4_xcat_nomotion_pilot/logs/sbatch_nik_f100_%j.log
# Task 4 NIK-F0 training at f100 (batch 16384 kept identical to f25 for a fair comparison).
set -e
cd /scratch/rnga/vvpshenov/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[job $SLURM_JOB_ID] host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date)"
$MM run -n torch29 python -u task4_nik.py --frac f100 --seed 0 --steps 40000
echo "[job $SLURM_JOB_ID] DONE $(date)"
