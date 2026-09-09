#!/bin/bash
# end-to-end test of the laptop loop: list two result dirs on helios
cd /net/beegfs/users/P101440/DCE_NIK || exit 1
echo "host $(hostname)  $(date '+%F %T')  user $USER"
for d in results/tofts_vs_patlak results/xcat_physical_motion_nik_vs_grasp; do
  echo; echo "== $d ($(find "$d" -type f 2>/dev/null | wc -l) files, $(du -sh "$d" 2>/dev/null | cut -f1))"
  ls -la "$d"
done
