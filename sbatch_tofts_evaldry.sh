#!/bin/bash
#SBATCH -J tevaldry
#SBATCH -p luna-gpu-short
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 1:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/results/tofts_vs_patlak/logs/evaldry_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK; P=/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python
$P -u tofts_eval_phantom.py --sim nomotion; $P -u tofts_eval_invivo.py; echo DRY_DONE
