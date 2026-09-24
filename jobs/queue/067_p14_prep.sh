#!/bin/bash
#SBATCH -J p14prep
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 120G
#SBATCH -t 8:00:00
# p14 (meas_topqmri_p14, liver slab) preparation chain for slices 21 / 24 / 27 (user choice 2026-09-24, liver + spleen + aorta rois):
# precompute (coil maps, grasp-pro all-spoke reference, radial bundle) -> per-dataset spoke masks -> nufft rulers -> model-free 31-spoke series
# -> aif gate -> liver / spleen roi proposal (for approval; nothing downstream uses it yet). all cpu.
set -uo pipefail
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=16 DCE_DS=p14
P="micromamba run -n torch29 python -u"; D=/net/beegfs/users/P101440/DCE_NIK
cd /net/beegfs/users/P101440/grasp_pro_py
for Z in 21 24 27; do [ -f results_ref_p14/slice_$Z.npz ] || $P precompute_ref.py --file /net/beegfs/users/P101440/dce_data/orig/meas_topqmri_p14.dat --out results_ref_p14 --slices $Z; done; echo "PRECOMPUTE exit $?"
cd $D
$P spoke_masks_build.py; echo "MASKS exit $?"
$P build_rulers.py 21,24,27; echo "RULERS exit $?"
for Z in 21 24 27; do $P step2_kidney.py $Z; done; echo "STEP2 exit $?"
for Z in 21 24 27; do $P aif_gate.py $Z; done; echo "AIF exit $?"
$P roi_propose_liver.py --slices 21,24,27; echo "ROI exit $?"
