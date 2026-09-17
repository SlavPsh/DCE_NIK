#!/bin/bash
# retrieval only: the two SENSE-B feasibility logs (check_b1_factor.py -> axchk_*.log, senseB_kernel_check.py -> sbk_*.log) from the DCE_NIK root
cd /net/beegfs/users/P101440/DCE_NIK; ls -la sbk_*.log axchk_*.log 2>/dev/null
for f in sbk_*.log axchk_*.log; do echo "===== $f"; cat "$f"; done 2>/dev/null
