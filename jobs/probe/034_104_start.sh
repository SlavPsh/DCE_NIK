#!/bin/bash
# retrieval only: did 104 start, warm-start lines of the first tasks
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.12j %.2t %.8M %R" -r | grep -E "JOBID|34260" | head -8
for f in $(ls -t jobs/log/104_sub16_tests_*.out 2>/dev/null | head -2); do echo "== $f"; grep -E "task|warmstart|model=|Traceback|Error|step " "$f" | tail -4 | cut -c1-180; done
