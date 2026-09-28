#!/bin/bash
# retrieval only: did 099 start
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.12j %.2t %.8M %R" -r | grep -E "JOBID|33913" | head -6
for f in $(ls -t jobs/log/099_p14_ks5_arms_*.out 2>/dev/null | head -2); do echo "== $f"; grep -E "task|model=|step " "$f" | tail -2 | cut -c1-160; done
