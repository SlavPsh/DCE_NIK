#!/bin/bash
#SBATCH -J b1factor
#SBATCH -p gpu
#SBATCH --gres gpu:1g.12gb:1
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 1:30:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/b1factor_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u check_b1_factor.py
