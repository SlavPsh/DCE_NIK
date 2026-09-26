#!/bin/bash
# retrieval only: did the grasp_pro_py fix arrive, and did the 091 render tasks start
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; git -C /net/beegfs/users/P101440/grasp_pro_py log --oneline -1; git log --oneline -1
squeue -u $USER -o "%.12i %.9P %.12j %.2t %.8M %R" -r | grep -E "JOBID|3387" | head -8
for f in $(ls -t jobs/log/091_p14_nik_render_*.out 2>/dev/null | head -2); do echo "== $f"; grep -vE "wandb|Warning|^\s*$" "$f" | tail -6 | cut -c1-160; done
