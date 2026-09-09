#!/bin/bash
#SBATCH -J v2cache
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 2:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/v2cache_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python} -u - <<'PY'
import sys; sys.path.insert(0,"/net/beegfs/users/P101440/DCE_NIK")
import xph_v2_sweep as S
kx,ky,kd,b1 = S.load_cached()
print("cache ready:", kx.shape, ky.shape, kd.shape, b1.shape)
PY
echo CACHE_DONE
