#!/bin/bash
# 047 array: logs of tasks 0-23 froze after probe 048 scancel. real state from slurm + output counts
squeue -j 3319010,3319011 -h -o "%i %T %M %N" | sort | head -40
sacct -j 3319010 -X -n -o JobID,State,Elapsed,ExitCode 2>/dev/null | grep -vE "RUNNING|PENDING" | head -30
for d in glam0.02 glam0.08 glam0.25; do echo "$d: $(ls /net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun/$d/slice_*.npy 2>/dev/null | wc -l) slices"; done
for t in 0 5 12 23; do f=/net/beegfs/users/P101440/DCE_NIK/jobs/log/047_gv2_rerun_glam2_recon_3319010_$t.out; echo "task $t log $(stat -c %y $f | cut -c12-19) $(wc -l < $f) lines: $(tail -1 $f | cut -c1-60)"; done
