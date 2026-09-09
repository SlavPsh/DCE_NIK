#!/bin/bash
#SBATCH --job-name=bt
#SBATCH -p defq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 10
#SBATCH --output=/net/beegfs/users/P101440/DCE_NIK/_bt.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u _bt.py
