#!/bin/bash
#SBATCH -J pkcoef
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 24G
#SBATCH -t 2:00:00
# phantom nik-tofts: ext-kety parameters fitted directly in the coefficient domain (no rendered frames) vs the image-domain fit vs truth
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=1 XPH_SIM=nomotion
micromamba run -n torch29 python -u pk_coef_fit_phantom.py --jobs 16
echo "PKCOEF exit $?"
