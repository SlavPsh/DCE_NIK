#!/bin/bash
# retrieval only: what the oom-killed 069 tasks left behind (checkpoints, arrays) and their peak host memory
cd /net/beegfs/users/P101440/DCE_NIK; date '+%T'
for d in results/tofts_vs_patlak/p14/invivo_prod/tofts8_sl21_s0 results/tofts_vs_patlak/p14/invivo_prod/free_sl21_s0 results/tofts_vs_patlak/p14/invivo_prod_oc/tofts8_sl21_s0; do echo "== $d"; ls -la --time-style=+%H:%M "$d" 2>/dev/null | awk '{print $5, $6, $7}'; ls "$d"/*/ 2>/dev/null | head; done
sacct -j 3380545,3381113,3387043 -o JobID%18,JobName%10,State%12,MaxRSS,ReqMem,Elapsed,ExitCode -P 2>/dev/null | grep -vE "extern" | head -20
grep -E "^(train|render|save|export)|MaxRSS|Killed" jobs/log/069_p14_nik_3380545.out | tail -5 | cut -c1-160
free -g | head -2; nproc
