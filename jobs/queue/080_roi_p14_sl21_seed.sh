#!/bin/bash
#SBATCH -J roiseed
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 0:30:00
# liver / spleen / aorta proposal on slice 21 with the aorta grown from the seed (row 145, col 137) read off the gridded decision figure
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=8 DCE_DS=p14
cd /net/beegfs/users/P101440/DCE_NIK; micromamba run -n torch29 python -u roi_propose_liver.py --slices 21 --aorta-seed "21:145,137"; echo "ROI exit $?"
