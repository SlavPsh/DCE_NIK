#!/bin/bash
#SBATCH -J huber
#SBATCH -p luna-gpu-short
#SBATCH --gres gpu:a100:1
#SBATCH -c 8
#SBATCH --mem 20G
#SBATCH -t 3:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/huber_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u 
echo "--- objective split across lam ---"
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u /tmp/huber_regime.py
