#!/bin/bash
# retrieval only: 109 progress
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.14j %.2t %.8M %R" -r | grep -E "JOBID|3441" | head -9
for f in $(ls -t jobs/log/109_sub16_own_protocol_34*.out 2>/dev/null); do echo "== $f"; grep -E "task [0-9]|step +[0-9]+ |TRAIN_DONE|Traceback|Error|oom" "$f" | tail -2 | cut -c1-140; done
