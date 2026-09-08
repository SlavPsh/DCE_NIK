#!/bin/bash
#SBATCH -J frontier
#SBATCH -p luna-gpu-short
#SBATCH --gres gpu:1g.10gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 2:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/frontier_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u xph_frontier.py
