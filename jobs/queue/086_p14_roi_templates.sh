#!/bin/bash
#SBATCH -J roitmpl
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:20:00
# painting templates for the manual p14 rois (model-free frames at 120 s and 60 s, 2x, per slice) -> rois_manual/ (png, synced by the agent)
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p14
cd /net/beegfs/users/P101440/DCE_NIK; micromamba run -n torch29 python -u roi_manual_export.py --slices 21,24,27; echo "TEMPLATES exit $?"
