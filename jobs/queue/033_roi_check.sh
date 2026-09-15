#!/bin/bash
#SBATCH -J roicheck
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 0:20:00
# roi masks (shared per slice) overlaid on every method's k80 image at t = 90 s, slices 21, 18, 19, plus mask sizes
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=4
micromamba run -n torch29 python -u roi_check_invivo.py --slices 21,18,19 --t-show 90
echo "ROI exit $?"
