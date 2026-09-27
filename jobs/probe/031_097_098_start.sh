#!/bin/bash
# retrieval only: did 097 / 098 start
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.10j %.2t %.8M %R" -r | grep -E "JOBID|33911" | head -8
for f in $(ls -t jobs/log/097_p14_prod_ks5_*.out jobs/log/098_p3_bandwidth_*.out 2>/dev/null | head -3); do echo "== $f"; grep -E "task|model=|step " "$f" | tail -2 | cut -c1-160; done
