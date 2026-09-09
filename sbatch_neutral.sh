#!/bin/bash
#SBATCH -J neutral
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 2:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/neutral_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u invivo_neutral_ruler.py
