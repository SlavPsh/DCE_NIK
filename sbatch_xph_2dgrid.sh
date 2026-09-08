#!/bin/bash
#SBATCH --job-name=xph-2dgrid
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/grid2d_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[grid2d $SLURM_JOB_ID] gpu=$CUDA_VISIBLE_DEVICES $(date)"
$MM run -n torch29 python -u xph_grasp_2dgrid.py
echo "[grid2d $SLURM_JOB_ID] DONE $(date)"
