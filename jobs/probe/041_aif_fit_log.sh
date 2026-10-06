#!/bin/bash
# retrieval only: head of the 114 task-0 log (prep output) and the aif_param files
cd /net/beegfs/users/P101440/DCE_NIK; sed -n 1,40p jobs/log/114_arrival_fixes_3469931.out | cut -c1-300
ls -la --time-style=+%H:%M aif_param_slice21.npz results/realdata_nik_vs_cs_figures/figures/aif_param* 2>&1 | awk '{print $5, $6, $7, $8, $9}'
git add results/realdata_nik_vs_cs_figures/figures/aif_param*.png 2>/dev/null && echo staged
