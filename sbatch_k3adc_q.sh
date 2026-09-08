#!/bin/bash
#SBATCH --job-name=k3adc-q
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:35:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/k3adcq_%j.log
export MAMBA_ROOT_PREFIX=/scratch/rnga/vvpshenov/micromamba
export K3A_QUICK=1
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u k3a_dc.py
