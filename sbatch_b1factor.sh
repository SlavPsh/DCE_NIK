#!/bin/bash
#SBATCH -J b1factor
#SBATCH -p luna-gpu-short
#SBATCH --gres gpu:1g.10gb:1
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 1:30:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/b1factor_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u check_b1_factor.py
