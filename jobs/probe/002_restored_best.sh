#!/bin/bash
# rank 8 basis study (log is gitignored) + early stop step of every in vivo tofts_vs_patlak run (old logs, not in git)
cd /net/beegfs/users/P101440/DCE_NIK || exit 1
R=results/tofts_vs_patlak
echo "== $R/basis_sl21_r8.log"; grep -vE '^\s*$' $R/basis_sl21_r8.log | tail -12
echo; echo "== restored best per run (old array logs: i = slice_idx*6 + model_idx*3 + seed; slices 18,19,21; patlak,tofts)"
for f in $R/logs/invivo_*.log; do
  printf '%-40s ' "$(basename $f)"; grep -oE 'restored best \(heldout [0-9.e+-]+\)' $f | tail -1 | tr -d '\n'
  printf '  best_step '; awk '/heldout [0-9]/{split($0,a,"heldout "); v=a[2]+0; s=$2+0; if (s>=2000 && v<b || b=="") {b=v; bs=s}} END{print bs}' $f
done
echo; echo "== heldout trajectory, slice 21 runs (i=12..17), every 8th eval"
for i in 12 13 14 15 16 17; do for f in $R/logs/invivo_*_$i.log; do echo "-- $(basename $f)"; grep -E 'heldout [0-9]' $f | awk 'NR<=2 || NR%8==0' | sed 's/^ *//'; done; done
