#!/bin/bash
#SBATCH -J tverify
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:40:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/verify_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python -u tofts_verify.py
