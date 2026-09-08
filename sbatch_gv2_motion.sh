#!/bin/bash
#SBATCH -J gv2mot
#SBATCH -p luna-gpu-short
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 2:00:00
#SBATCH -o /scratch/rnga/vvpshenov/DCE_NIK/results/tofts_vs_patlak/logs/gv2_motion_%j.log
cd /scratch/rnga/vvpshenov/DCE_NIK; export XPH_SIM=motion
/scratch/rnga/vvpshenov/micromamba/envs/torch29/bin/python -u xph_v2_sweep.py 5; echo GV2_MOTION_DONE
