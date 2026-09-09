#!/bin/bash
#SBATCH -J dcesmoke
#SBATCH -p gpu
#SBATCH --gres gpu:1g.12gb:1
#SBATCH -c 2
#SBATCH --mem 8G
#SBATCH -t 0:20:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/dcesmoke_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u dcenet_smoke.py
