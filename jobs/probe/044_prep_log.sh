#!/bin/bash
# why the 043 prep log was not committed: size, head, agent log lines, prep output dir
D=/net/beegfs/users/P101440/DCE_NIK; O=/net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun
ls -la $D/jobs/log/043_gv2_rerun_3308108.out 2>&1; wc -c $D/jobs/log/043_gv2_rerun_3308108.out 2>&1
grep -vE "^\s+slice [0-9]|libmamba|Waiting" $D/jobs/log/043_gv2_rerun_3308108.out 2>&1 | head -40
echo "== agent log"; grep -hE "043|skip large|FAILED" $D/logs/agent_*.out 2>/dev/null | tail -8
echo "== out"; ls $O; du -sh $O/prep $O/nufft $O/lam* 2>&1; cat $O/geom.json 2>&1 | head -30
squeue -u P101440 -h -o "%i %j %T %M" | sort | uniq -c | sort -rn | head -6
