#!/bin/bash
#SBATCH --job-name=t4c-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --array=0-8
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/task4c_nik_capacity_audit/slurm/train_%A_%a.log
# 3 widths x 3 seeds. idx -> width=[256,512,768][idx/3], seed=[0,1,2][idx%3].
set -e
cd /net/beegfs/users/P101440/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/net/beegfs/users/P101440/micromamba/bin/micromamba
WIDTHS=(256 512 768); SEEDS=(0 1 2)
W=${WIDTHS[$((SLURM_ARRAY_TASK_ID/3))]}; S=${SEEDS[$((SLURM_ARRAY_TASK_ID%3))]}
echo "[job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] width=$W seed=$S host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES $(date)"
$MM run -n torch29 python -u task4c_nik.py --hidden-width $W --seed $S --steps 40000 --ckpt-every 2000
echo "[job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] DONE $(date)"
