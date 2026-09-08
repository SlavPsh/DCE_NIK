#!/bin/bash
#SBATCH -J tverify
#SBATCH -p luna-gpu-short
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:40:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/results/tofts_vs_patlak/verify_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK
/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python -u tofts_verify.py
