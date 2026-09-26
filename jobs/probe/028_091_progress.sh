#!/bin/bash
# retrieval only: 091 render tasks, live memory, outputs so far
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.12j %.2t %.8M %R" -r | grep -E "JOBID|3387" | head -6
for j in 3387047 3387048; do sstat -j $j.batch -o JobID,MaxRSS,AveRSS -P 2>/dev/null | tail -1; done
for f in $(ls -t jobs/log/091_p14_nik_render_*.out 2>/dev/null | head -4); do echo "== $f"; grep -vE "wandb|Warning|^\s*$" "$f" | tail -3 | cut -c1-160; done
ls -la --time-style=+%H:%M results/tofts_vs_patlak/p14/invivo_prod/tofts8_sl21_s0 results/tofts_vs_patlak/p14/invivo_prod/tofts8_sl21_s1 2>/dev/null | awk '{print $5, $6, $7}'
