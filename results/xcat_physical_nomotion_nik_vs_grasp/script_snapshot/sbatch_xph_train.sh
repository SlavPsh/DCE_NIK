#!/bin/bash
#SBATCH --job-name=xph-train
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/train_%x_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
echo "[train $SLURM_JOB_ID] W=$W KS=$KS SEED=$SEED gpu=$CUDA_VISIBLE_DEVICES $(date)"
$MM run -n torch29 python -u xph_train.py --hidden-width $W --k-sigma $KS --seed $SEED --steps 40000 --ckpt-every 2000
$MM run -n torch29 python -u xph_eval.py --tag w${W}_ks${KS}_s${SEED}
echo "[train $SLURM_JOB_ID] DONE $(date)"
