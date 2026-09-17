#!/bin/bash
#SBATCH -J mfpeak
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 1:00:00
# (a) model-free first-pass peak vs sliding-window width (does the 31-spoke reference under-read the peak?), slices 21, 18, 19; (b) body support masks on the nx grid for the k-space support prior
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=4
P="micromamba run -n torch29 python -u"
$P support_mask.py --slices 21,18,19 --dilate 4; echo "MASK exit $?"
for Z in 21 18 19; do $P mf_peak_check.py --slice $Z --windows 31,21,15,11,7; done; echo "PEAK exit $?"
