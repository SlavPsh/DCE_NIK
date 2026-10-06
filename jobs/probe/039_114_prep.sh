#!/bin/bash
# retrieval only: did the 114 prep (aif fit + bases) succeed, did the tasks start
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.12i %.12j %.2t %.8M %R" -r | grep -E "JOBID|34699" | head -9
f=$(ls -t jobs/log/114_arrival_fixes_*.out 2>/dev/null | head -3); for x in $f; do echo "== $x"; grep -E "task [0-9]|fit rms|AIF_PARAM|PREP exit|SELECTED|wrote|model=|prior groups|Traceback|Error|step +1000 " "$x" | tail -6 | cut -c1-200; done
tail -3 results/tofts_vs_patlak/basis_sl21_r8_param.log results/tofts_vs_patlak/basis_sl21_r5.log 2>/dev/null | cut -c1-160
git add results/realdata_nik_vs_cs_figures/figures/aif_param_slice21.png 2>/dev/null && echo "fig staged"
