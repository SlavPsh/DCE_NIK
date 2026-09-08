#!/bin/bash
#SBATCH --job-name=envsweep
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2:30:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/envsweep_%x_%j.log
set -e; cd /scratch/rnga/vvpshenov/DCE_NIK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=/home/rnga/vvpshenov/my-scratch/micromamba/bin/micromamba
TAG=sub16_e${ENVCODE}_w768_s${SEED}
echo "[envsweep $SLURM_JOB_ID] ENV=$ENV TAG=$TAG $(date)"
$MM run -n torch29 python -u xph_img_train.py --model wire_ff_subspace --rank 16 --warmstart --width 768 --seed $SEED --env $ENV --runtag $TAG --steps 40000 --ckpt-every 2000
$MM run -n torch29 python -u xph_img_eval.py --tag $TAG --model wire_ff_subspace --rank 16 --width 768 --env $ENV --save-prefix envsweep
echo "[envsweep $SLURM_JOB_ID] DONE $TAG $(date)"
