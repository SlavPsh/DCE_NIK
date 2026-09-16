#!/bin/bash
# 043 slice array: 14 tasks stopped writing at ~18:55. diagnose, cancel only RUNNING tasks whose log is stale > 30 min, and the post job waiting on them
D=/net/beegfs/users/P101440/DCE_NIK
echo "== squeue"; squeue -j 3308510,3308511 -h -o "%i %T %M %R %N" 2>&1 | head -20
echo "== sacct"; sacct -j 3308510 -X -n -o JobID,State,Elapsed,ExitCode,NodeList 2>&1 | grep -vE "COMPLETED" | head -20
echo "== stale logs (mmin +30)"; find $D/jobs/log -name "043_gv2_rerun_recon_3308510_*.out" -mmin +30 -exec sh -c 'grep -q SLICE_DONE "$1" || echo "$1 $(date -r "$1" +%H:%M) $(tail -c 60 "$1" | tr "\n" " ")"' _ {} \;
echo "== cancel"
for f in $(find $D/jobs/log -name "043_gv2_rerun_recon_3308510_*.out" -mmin +30); do grep -q SLICE_DONE "$f" && continue; t=${f##*_}; t=${t%.out}; st=$(squeue -h -j 3308510_$t -o %T 2>/dev/null); [ "$st" = "RUNNING" ] && { scancel 3308510_$t && echo "cancelled task $t"; }; done
scancel 3308511 && echo "cancelled post 3308511"
echo "== agent"; tail -3 $D/logs/agent_3298317.out 2>/dev/null; squeue -n helios-agent -h -o "%i %T %M %L"
echo "== outputs"; for d in nufft lam0.02 lam0.08 lam0.25; do echo "$d: $(ls /net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun/$d/slice_*.npy 2>/dev/null | wc -l) slices"; done
