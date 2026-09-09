#!/bin/bash
#SBATCH --job-name=xph-img
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:30:00
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/img_%x_%j.log
set -e; cd /net/beegfs/users/P101440/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/net/beegfs/users/P101440/micromamba/bin/micromamba
echo "[img $SLURM_JOB_ID] MODEL=$MODEL RANK=$RANK WS=$WS W=$W SEED=$SEED $(date)"
WSFLAG=""; [ "$WS" = "1" ] && WSFLAG="--warmstart"
SHORT=$([ "$MODEL" = "wire_ff_subspace" ] && echo "sub${RANK}" || echo "free")
$MM run -n torch29 python -u xph_img_train.py --model $MODEL --rank $RANK $WSFLAG --width $W --seed $SEED --steps 40000 --ckpt-every 2000
$MM run -n torch29 python -u xph_img_eval.py --tag ${SHORT}_w${W}_s${SEED} --model $MODEL --rank $RANK --width $W
echo "[img $SLURM_JOB_ID] DONE $(date)"
