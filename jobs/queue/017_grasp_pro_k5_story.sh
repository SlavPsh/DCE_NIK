#!/bin/bash
#SBATCH -J prok5b
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 1:30:00
# grasp pro on the phantom through the nufft pathway (report code) at K 5, 25 spokes/frame (68 frames), same 5 of 7 spokes; then the story figures with it (tag _prok5) and the in vivo panel with curve nrmse
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH XPH_SIM=nomotion OMP_NUM_THREADS=8
micromamba run -n torch29 python -u xph_grasp_pro_k5.py --K 5 --G 5; echo "PRO exit $?"
micromamba run -n torch29 python -u story_figs.py --pro k5g5 --tag _prok5 --t-phantom 90 --t-invivo 90 --tofts tofts; echo "STORY exit $?"
