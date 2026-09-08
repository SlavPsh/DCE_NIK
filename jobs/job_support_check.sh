#!/bin/bash
#SBATCH --job-name=support-chk
#SBATCH --partition=luna-gpu-short
#SBATCH --gres=gpu:2g.20gb:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:30
#SBATCH --output=/scratch/rnga/vvpshenov/DCE_NIK/logs/slurm-%x-%j.out
#SBATCH --error=/scratch/rnga/vvpshenov/DCE_NIK/logs/slurm-%x-%j.out
set -uo pipefail
export PATH="/scratch/rnga/vvpshenov/micromamba/bin:$PATH"
export MAMBA_ROOT_PREFIX="/scratch/rnga/vvpshenov/micromamba"
micromamba run -n torch29 python /scratch/rnga/vvpshenov/DCE_NIK/render_support_check.py
