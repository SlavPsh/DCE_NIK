#!/bin/bash
#SBATCH --job-name=ocreal
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/realdata_nik_vs_cs_figures/ocreal_%j.log
export MAMBA_ROOT_PREFIX=/scratch/rnga/vvpshenov/micromamba
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u outcoil_real.py --coilmode ${MODE:-output} --steps ${STEPS:-30000}
