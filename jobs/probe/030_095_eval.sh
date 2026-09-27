#!/bin/bash
# retrieval only: state of the 095 eval chain
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.14j %.2t %.8M %R" -r | head -8
sacct -S 2026-09-26T19:00 -o JobID%14,JobName%14,State%12,Elapsed,Reason%30 -P 2>/dev/null | grep -E "p14bw_eval|JobID" | head -5
for f in $(ls -t jobs/log/095_p14_bandwidth_eval_*.out 2>/dev/null | head -1); do echo "== $f"; grep -vE "^\s*$" "$f" | tail -8 | cut -c1-200; done
ls results/tofts_vs_patlak/p14/bw_test/*/nik_slice_24_cplx.npy 2>/dev/null
