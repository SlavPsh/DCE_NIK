#!/bin/bash
#SBATCH -J iqnoise
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 1:30:00
# image-quality tracker with the added spatial-noise / high-pass / edge-sharpness numbers on the final recons and the 044 snapshots
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=4
IQ_VARIANTS=rms1_wd3e-3,rms1_wd1e-2 IQ_TAG=_trade2 micromamba run -n torch29 python -u tofts_iq_track.py --slice 21
echo "IQ2 exit $?"
