#!/bin/bash
# array inventory for the story figures: phantom arrays, v2 sweep, in vivo k80 nik runs, outcoil (sub16) files
cd /net/beegfs/users/P101440/DCE_NIK || exit 1
O=results/xcat_physical_nomotion_nik_vs_grasp
echo "== $O/arrays"; ls -la $O/arrays | awk '{print $5, $9}' | sort -k2 | head -80
echo "== $O/v2_sweep npy"; ls $O/v2_sweep/*.npy
echo "== results_sl21_k80"; ls -la results_sl21_k80 | awk '{print $5, $9}'
echo "== outcoil files (sub16 k80 candidates)"; ls -la results/realdata_nik_vs_cs_figures/outcoil_*.npy 2>/dev/null | awk '{print $5, $9}'
echo "== invivo_k80 runs"; ls results/tofts_vs_patlak/invivo_k80/
echo "== grasp refs k80 sl21"; ls -la /net/beegfs/users/P101440/grasp_v2/results_grasp_v2/gv2_slice21_n12_k80.npy /net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs/cs_slice21_f80match.npy | awk '{print $5, $9}'
echo "== outcoil_real.py usage notes"; sed -n '1,20p' outcoil_real.py | grep -iE 'k80|keep|usage|python' | head -6
