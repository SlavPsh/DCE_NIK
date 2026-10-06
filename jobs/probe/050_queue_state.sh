#!/bin/bash
# retrieval only: queue state 118 / 119 / 120
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.14j %.2t %.9M %R" -r | head -14
for f in $(ls -t jobs/log/118_delay_prior_34*.out jobs/log/119_k100_34*.out jobs/log/120_free_tsigma_34*.out 2>/dev/null | head -4); do echo "== $f"; grep -E "task [0-9]|step +[0-9]+ |TRAIN_DONE|Traceback" "$f" | tail -2 | cut -c1-120; done
