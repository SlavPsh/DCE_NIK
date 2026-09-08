#!/bin/bash
#SBATCH -J teval
#SBATCH -p luna-gpu-short
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 3:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/results/tofts_vs_patlak/logs/eval_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK; P=/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python
echo "### phantom nomotion"; $P -u tofts_eval_phantom.py --sim nomotion
echo "### phantom motion"; $P -u tofts_eval_phantom.py --sim motion
echo "### in vivo"; $P -u tofts_eval_invivo.py
echo EVAL_ALL_DONE
