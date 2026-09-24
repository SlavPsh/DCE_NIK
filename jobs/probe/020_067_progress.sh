#!/bin/bash
# retrieval only: tail of the running 067 log + what results_ref_p14 already holds
cd /net/beegfs/users/P101440/DCE_NIK; f=$(ls -t jobs/log/067_p14_prep_*.out 2>/dev/null | head -1); echo "log $f"; grep -v "^\s*$" "$f" | tail -25 | cut -c1-200
ls -la --time-style=+%H:%M /net/beegfs/users/P101440/grasp_pro_py/results_ref_p14/ 2>/dev/null | awk '{print $5, $6, $7}'; ls --time-style=+%H:%M -la step2_p14_slice*.npz aif_p14_slice*.npz results_nufft_p14_slice*/meta.json 2>/dev/null | awk '{print $6, $7}'; date '+%T'
