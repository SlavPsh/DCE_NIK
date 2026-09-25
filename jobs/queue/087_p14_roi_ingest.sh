#!/bin/bash
#SBATCH -J roiingest
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:20:00
# manual p14 rois: outlines painted on the templates (red liver, green spleen, blue aorta) -> filled masks, eroded 2 px, approved roi files + overlay figures with curves
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH DCE_DS=p14
cd /net/beegfs/users/P101440/DCE_NIK; micromamba run -n torch29 python -u roi_manual_ingest.py --slices 21,24,27 --erode 2; echo "INGEST exit $?"
