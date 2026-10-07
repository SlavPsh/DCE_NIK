#!/bin/bash
# retrieval only: did the k100 campaign start cleanly (123 grasp, 124 p3 arms, 125 p14 arms, 126 eval wait loop)
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.10j %.2t %.8M %R" -r | grep -E "JOBID|34825" | head -12
for f in $(ls -t jobs/log/124_k100_p3_34*.out jobs/log/123_grasp_allspokes_34*.out jobs/log/126_k100_eval_34*.out 2>/dev/null | head -4); do echo "== $f"; grep -E "task [0-9]|model=|step +[0-9]+ |SAVED|cached|waiting|Traceback|Error|exists" "$f" | tail -3 | cut -c1-160; done
