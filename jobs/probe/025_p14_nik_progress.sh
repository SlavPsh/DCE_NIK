#!/bin/bash
# retrieval only: squeue state of 069 array, per-task train_done lines, and finished p14 run dirs
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'; squeue -u $USER -o "%.10i %.9P %.28j %.2t %.10M %R" -r | grep -E "JOBID|3380545" | head -30
f=$(ls -t jobs/log/069_p14_nik_*.out 2>/dev/null | head -1); echo "log $f"; grep -E "TRAIN_DONE|EVAL|Traceback|exit [1-9]|CUDA out" "$f" | tail -25 | cut -c1-200; grep -E "step +[0-9]+/" "$f" | tail -2 | cut -c1-160
ls -d results/tofts_vs_patlak/invivo_k80_p14*/* 2>/dev/null; ls results/tofts_vs_patlak/*p14* 2>/dev/null
