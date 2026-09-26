#!/bin/bash
# retrieval only: did the 095 bandwidth tasks start
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.12j %.2t %.8M %R" -r | grep -E "JOBID|33870" | head -8
for f in $(ls -t jobs/log/095_p14_bandwidth_*.out 2>/dev/null | head -2); do echo "== $f"; grep -E "task|model=|step " "$f" | tail -3 | cut -c1-160; done
