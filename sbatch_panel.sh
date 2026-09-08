#!/bin/bash
#SBATCH -J panel
#SBATCH -p luna-gpu-short
#SBATCH --gres gpu:2g.20gb:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 1:30:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/panel_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
${PY:-/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python} -u xph_image_metrics.py
