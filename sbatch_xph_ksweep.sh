#!/bin/bash
#SBATCH --job-name=xph-ksweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:30:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/ksweep_%j.log
set -e; cd /net/beegfs/users/P101440/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/net/beegfs/users/P101440/micromamba/bin/micromamba
echo "[ksweep $SLURM_JOB_ID] gpu=$CUDA_VISIBLE_DEVICES $(date)"
$MM run -n torch29 python -u xph_grasp_ksweep.py
echo "[ksweep $SLURM_JOB_ID] DONE $(date)"
