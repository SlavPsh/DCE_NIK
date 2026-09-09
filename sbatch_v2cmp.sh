#!/bin/bash
#SBATCH -J v2cmp
#SBATCH -p gpu
#SBATCH --gres gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 2:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/v2cmp_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u xph_v2_vs_nik.py
