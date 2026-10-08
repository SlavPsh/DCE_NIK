#!/bin/bash
# queue state after submitting 133 (phantom per-atom wd): 130 progress, 133 pending on the gpu quota?
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'
squeue -u $USER -o "%.12i %.10j %.2t %.8M %R" -r | head -20
grep -hE "TRAIN_DONE|Traceback|step 10000" results/tofts_vs_patlak/peak_levers/*/tofts8_sl21_s*/train.log 2>/dev/null | cut -c1-100
for f in results/tofts_vs_patlak/peak_levers/*/tofts8_sl21_s*/train.log; do echo "$f: $(grep -cE '^ *step' $f) step lines, last: $(grep -E '^ *step' $f | tail -1 | cut -c1-70)"; done
