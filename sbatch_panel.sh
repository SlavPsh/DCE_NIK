#!/bin/bash
#SBATCH -J panel
#SBATCH -p gpu
#SBATCH --gres gpu:2g.24gb:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 1:30:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/panel_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u xph_image_metrics.py
