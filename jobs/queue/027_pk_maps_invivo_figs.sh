#!/bin/bash
#SBATCH -J pkfigs5
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:30:00
# re-render the in vivo pk map figures from the saved fits, masked to voxels that enhance in the reference
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
micromamba run -n torch29 python -u pk_maps_invivo.py --slices 21,18,19 --figs-only
echo "PK exit $?"
