#!/bin/bash
#SBATCH -J newrois
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 1:00:00
# approved anatomical rois (rois_proposed_sl<Z>.npz, wired into consolidated.slice_ctx): overlay check, span diagnostic and in vivo story panels redone with them
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=4
P="micromamba run -n torch29 python -u"
$P roi_check_invivo.py --slices 21,18,19 --t-show 90; echo "ROI exit $?"
for Z in 21 18 19; do $P tofts_span_diag.py --slice $Z; done; echo "SPAN exit $?"
$P story_figs.py --only invivo --t-invivo 90 --tofts tofts; echo "STORY exit $?"
