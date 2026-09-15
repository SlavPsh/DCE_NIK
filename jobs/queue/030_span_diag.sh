#!/bin/bash
#SBATCH -J spandiag
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 0:30:00
# in vivo pk-arm amplitude deficit: model-free curves projected onto the tofts atoms at ranks 3/5/8/12 vs the trained arms, slices 21, 18, 19
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=4
for Z in 21 18 19; do micromamba run -n torch29 python -u tofts_span_diag.py --slice $Z; done
echo "SPAN exit $?"
