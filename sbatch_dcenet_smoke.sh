#!/bin/bash
#SBATCH -J dcesmoke
#SBATCH -p luna-gpu-short
#SBATCH --gres gpu:1g.10gb:1
#SBATCH -c 2
#SBATCH --mem 8G
#SBATCH -t 0:20:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/dcesmoke_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u dcenet_smoke.py
