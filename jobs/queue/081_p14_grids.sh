#!/bin/bash
#SBATCH -J p14grid
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 0:40:00
# gridded decision figures for slices 24, 27 (aorta candidates + grid) and the seeded proposal on slice 21 (aorta seed 145,137; liver / spleen auto) with gridded axes
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=8 DCE_DS=p14
cd /net/beegfs/users/P101440/DCE_NIK; P="micromamba run -n torch29 python -u"
for Z in 24 27; do $P roi_candidates_fig.py --slice $Z; done; echo "CAND exit $?"
$P roi_propose_liver.py --slices 21 --aorta-seed "21:145,137"; echo "ROI exit $?"
