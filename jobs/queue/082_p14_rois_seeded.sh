#!/bin/bash
#SBATCH -J p14rois
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 0:40:00
# liver / spleen / aorta proposal for p14 slices 21, 24, 27 from per-slice seeds read off the gridded figures (for approval)
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=8 DCE_DS=p14
cd /net/beegfs/users/P101440/DCE_NIK
micromamba run -n torch29 python -u roi_propose_liver.py --slices 21,24,27 --aorta-seed "21:145,137;24:140,134;27:140,133" --liver-seed "21:110,80;24:140,70;27:120,70" --spleen-seed "21:150,200;24:150,215;27:160,215"; echo "ROI exit $?"
