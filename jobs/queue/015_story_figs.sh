#!/bin/bash
#SBATCH -J story2
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 0:40:00
# presentation story figures: 4 phantom panels (one per nik family, same 5 of 7 spokes/frame) + in vivo k80 panel (4 nik + grasp pro + grasp)
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH XPH_SIM=nomotion
micromamba run -n torch29 python -u story_figs.py --t-phantom 90 --t-invivo 90 --tofts tofts
echo "STORY exit $?"
