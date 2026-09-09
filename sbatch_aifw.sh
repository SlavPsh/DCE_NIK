#!/bin/bash
#SBATCH -J aifw
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 16G
#SBATCH -t 2:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/aifw_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u truth_aif_width.py
