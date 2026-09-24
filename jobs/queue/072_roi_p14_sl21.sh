#!/bin/bash
#SBATCH -J roip14
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 0:30:00
# liver / spleen roi proposal v2 (spleen rule: >= liver-level late enhancement, faster ratio than the liver, 3-px opening) on the finished slice 21
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=8 DCE_DS=p14
cd /net/beegfs/users/P101440/DCE_NIK; micromamba run -n torch29 python -u roi_propose_liver.py --slices 21; echo "ROI exit $?"
