#!/bin/bash
# retrieval only: k100 eval state
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.9j %.2t %.9M %R" -r | grep -E "k100|JOBID" | awk '{print $1, $2, $3, $4, $5}' | head -5
echo "p14 outputs: $(ls results/tofts_vs_patlak/p14/invivo_k100/*/nik_slice_*_cplx.npy results/tofts_vs_patlak/p14/invivo_k100_oc/*/nik_slice_*_cplx.npy 2>/dev/null | wc -l) / 21"
f=$(ls -t jobs/log/126_k100_eval_*.out | head -1); grep -E "all inputs|waiting|EVAL|FIGS|GIFS|DONE|TIMEOUT|Traceback|Error|saved" "$f" | tail -8 | cut -c1-140
