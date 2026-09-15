#!/bin/bash
#SBATCH -J aortaoff
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:20:00
# phantom aorta washout offset: partial volume vs regional low-|k| error vs temporal model (erosion series, ring error, late error maps)
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=4 XPH_SIM=nomotion
micromamba run -n torch29 python -u phantom_aorta_offset.py
echo "AORTA exit $?"
