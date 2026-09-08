#!/bin/bash
#SBATCH --job-name=Af10
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/Afull_f10_%j.log
export MAMBA_ROOT_PREFIX=/scratch/rnga/vvpshenov/micromamba
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u sense_forward_train.py --steps 20000 --fbatch 6 --rank 16 --seed 0 --tag _f10 --w0 10 --s0 5 --imsig 2
