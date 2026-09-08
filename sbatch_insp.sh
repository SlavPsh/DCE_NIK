#!/bin/bash
#SBATCH --job-name=insp
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp/slurm/insp_%j.log
export MAMBA_ROOT_PREFIX=/scratch/rnga/vvpshenov/micromamba
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u inspect_senseA.py
