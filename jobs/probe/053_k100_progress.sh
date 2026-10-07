#!/bin/bash
# retrieval only: k100 campaign progress (outputs on disk, queue)
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; echo "p3 outputs: $(ls results/tofts_vs_patlak/invivo_k100/*/nik_slice_*_cplx.npy results/tofts_vs_patlak/invivo_k100_oc/*/nik_slice_*_cplx.npy 2>/dev/null | wc -l) / 21 (5 from 119)"; echo "p14 outputs: $(ls results/tofts_vs_patlak/p14/invivo_k100/*/nik_slice_*_cplx.npy results/tofts_vs_patlak/p14/invivo_k100_oc/*/nik_slice_*_cplx.npy 2>/dev/null | wc -l) / 21"
squeue -u $USER -o "%.12i %.9j %.2t %.9M %R" -r | grep -E "k100|JOBID" | awk '{print $1, $2, $3, $4, $5}' | head -6; echo "running: $(squeue -u $USER -h -t R -r | grep -c k100p), pending: $(squeue -u $USER -h -t PD -r | grep -c k100p)"
