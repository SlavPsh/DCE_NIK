#!/bin/bash
# grasp_pro_py and grasp_v2 checkouts on helios have no upstream, so the agent's pull fails every cycle (probe 005)
for r in grasp_pro_py grasp_v2 DCE_NIK; do
  cd /net/beegfs/users/P101440/$r || continue
  git branch --set-upstream-to=origin/main main 2>&1 | tail -1
  git pull --rebase --autostash -q origin main && echo "$r: $(git log --oneline -1)"
done
grep -n 'SLICE = ' /net/beegfs/users/P101440/grasp_pro_py/cs_nikmatch.py; grep -n '_SL = ' /net/beegfs/users/P101440/grasp_v2/grasp_v2_real.py
