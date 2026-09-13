#!/bin/bash
#SBATCH -J storyiv
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 0:30:00
# in vivo story panels per nik family (reference | nik | grasp pro | grasp), all on k80, plus the combined panel
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
micromamba run -n torch29 python -u story_figs.py --only invivo --t-invivo 90 --tofts tofts
echo "STORY exit $?"
