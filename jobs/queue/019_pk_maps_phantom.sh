#!/bin/bash
#SBATCH -J pkmaps2
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 32G
#SBATCH -t 4:00:00
# fitted ext-kety parameter maps on the phantom for truth, nik tofts/free/patlak, grasp pro K5, grasp; DCE-NET curve_fit per enhancing voxel
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH XPH_SIM=nomotion OMP_NUM_THREADS=1
micromamba run -n torch29 python -u pk_maps_phantom.py --jobs 16
echo "PK exit $?"
