#!/bin/bash
# why did 006 run slice 21: repo state of grasp_pro_py / grasp_v2 on helios, and env propagation through micromamba run
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
for r in grasp_pro_py grasp_v2; do cd /net/beegfs/users/P101440/$r || continue; echo "== $r: $(git log --oneline -1) | status: $(git status --short | wc -l) dirty | behind: $(git fetch -q origin; git rev-list --count HEAD..origin/main)"; done
grep -n 'SLICE = ' /net/beegfs/users/P101440/grasp_pro_py/cs_nikmatch.py; grep -n '_SL = ' /net/beegfs/users/P101440/grasp_v2/grasp_v2_real.py
echo "== env through micromamba run:"; SLICE=18 micromamba run -n torch29 python -c "import os; print('SLICE =', os.environ.get('SLICE'))"
echo "== agent log tail (pull failures?)"; grep -h 'FAILED' /net/beegfs/users/P101440/DCE_NIK/logs/agent_*.out 2>/dev/null | tail -5
