#!/bin/bash
#SBATCH -J aortagrid
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 0:30:00
# aorta decision figures for p14 (numbered early-enhancing blobs + curves), every slice whose model-free series exists
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=8 DCE_DS=p14
cd /net/beegfs/users/P101440/DCE_NIK
for Z in 21 24 27; do [ -f step2_p14_slice$Z.npz ] && [ -f results_nufft_p14_slice$Z/meta.json ] && micromamba run -n torch29 python -u roi_candidates_fig.py --slice $Z || echo "slice $Z not ready"; done; echo "CAND exit $?"
