#!/bin/bash
#SBATCH --job-name=nik-grasp
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-07:00
#SBATCH --nice=10000
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/slurm-%x-%j.out
#SBATCH --error=/scratch/rnga/vvpshenov/DCE_NIK/slurm-%x-%j.err

export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
eval "$(/scratch/rnga/vvpshenov/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate torch29

nvidia-smi || true
cd /scratch/rnga/vvpshenov/DCE_NIK

python train_grasp_nik.py \
  --slices all \
  --steps 8000 \
  --save-dir /scratch/rnga/vvpshenov/grasp_pro_py/results_nik
echo "EXIT $?"
