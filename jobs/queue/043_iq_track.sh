#!/bin/bash
#SBATCH -J iqtrack
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 1:30:00
# image quality (haarpsi / ssim / psnr vs grasp-pro all-spoke anatomy, air energy, temporal noise) vs curve fidelity, per training step and per method, slice 21
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=4
micromamba run -n torch29 python -u tofts_iq_track.py --slice 21
echo "IQ exit $?"
