#!/bin/bash
#SBATCH -J huber
#SBATCH -p gpu
#SBATCH --gres gpu:h100:1
#SBATCH -c 8
#SBATCH --mem 20G
#SBATCH -t 3:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/huber_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u 
echo "--- objective split across lam ---"
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u /tmp/huber_regime.py
