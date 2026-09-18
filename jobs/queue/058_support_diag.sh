#!/bin/bash
#SBATCH -J supdiag
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:40:00
# support prior geometry check: the trainer's prior render vs the production render, energy inside / outside the mask, sup3 vs base model
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
AT=/net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/amp_track
micromamba run -n torch29 python -u support_diag.py --slice 21 --runs base:$AT/rms1_wd3e-3_sl21,sup3:$AT/sup3_sl21; echo "DIAG exit $?"
