#!/bin/bash
#SBATCH -J roiprop
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 0:20:00
# proposed anatomical kidney rois from the model-free series, overlay figures for approval (nothing uses them yet), slices 21, 18, 19
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=4
micromamba run -n torch29 python -u roi_propose_invivo.py --slices 21,18,19 --thr 0.35 --erode 1
echo "ROIPROP exit $?"
