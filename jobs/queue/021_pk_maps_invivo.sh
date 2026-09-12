#!/bin/bash
#SBATCH -J pkinvivo
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 32G
#SBATCH -t 3:00:00
# quick in vivo ext-kety maps at k80: literature T10, model-free aorta aif, DCE-NET fit per body voxel, slices 21, 18, 19
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=1
micromamba run -n torch29 python -u pk_maps_invivo.py --slices 21,18,19 --jobs 16
echo "PK exit $?"
