#!/bin/bash
#SBATCH -J p14basis
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem 96G
#SBATCH -t 6:00:00
# p14 stage 2 (after 067): tofts basis per slice (rank 8, unit-rms copy), support masks (body dilated 6 px on the 512 grid), grasp-pro f80match and
# grasp v2 n12 k80 references on the same k80 views. every step gated on its inputs from 067.
set -uo pipefail
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=16 DCE_DS=p14
P="micromamba run -n torch29 python -u"; D=/net/beegfs/users/P101440/DCE_NIK; RES=$D/results/tofts_vs_patlak
cd $D
for Z in 21 24 27; do $P aif_gate.py $Z; done; echo "AIF (approved aorta) exit $?"
for Z in 21 24 27; do
  [ -f $D/aif_p14_slice$Z.npz ] || { echo "aif for slice $Z missing (067 not done)"; exit 1; }
  [ -f $RES/basis_p14_sl${Z}_r8.npz ] || $P nik_tofts_basis.py --aif $D/aif_p14_slice$Z.npz --out $RES/basis_p14_sl${Z}_r8_tmp.npz --ranks 8 > $RES/basis_p14_sl${Z}_r8.log 2>&1 && mv -f $RES/basis_p14_sl${Z}_r8_tmp.npz $RES/basis_p14_sl${Z}_r8.npz 2>/dev/null; tail -2 $RES/basis_p14_sl${Z}_r8.log
done
$P basis_rms1.py $RES/basis_p14_sl21_r8.npz $RES/basis_p14_sl24_r8.npz $RES/basis_p14_sl27_r8.npz; echo "BASIS exit $?"
$P support_mask.py --slices 21,24,27 --dilate 6; echo "SUPPORT exit $?"
cd /net/beegfs/users/P101440/grasp_pro_py
for Z in 21 24 27; do echo "== grasp pro f80match slice $Z $(date '+%T')"; SLICE=$Z $P cs_nikmatch.py | grep -E 'slice|SAVED|Error|Traceback'; done; echo "GP exit $?"
cd /net/beegfs/users/P101440/grasp_v2
echo "== grasp v2 n12 k80 slices 21,24,27 $(date '+%T')"; SET=sweep NLINE=12 KEEP80=1 LAM_FRAC=0.25 SLICES=21,24,27 $P grasp_v2_real.py | grep -E 'SAVED|cached|DONE|Error|Traceback'; echo "GV exit $?"
ls -la /net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14/ /net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14/ 2>/dev/null | awk '{print $5, $9}'
