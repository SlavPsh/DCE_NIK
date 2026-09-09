#!/bin/bash
#SBATCH -J gv2mot
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 8
#SBATCH --mem 40G
#SBATCH -t 2:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/logs/gv2_motion_%j.log
cd /net/beegfs/users/P101440/DCE_NIK; export XPH_SIM=motion
/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python -u xph_v2_sweep.py 5; echo GV2_MOTION_DONE
