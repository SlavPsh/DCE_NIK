#!/bin/bash
#SBATCH -J sub16atoms
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 1:00:00
# atom diagnostic of the 104 sub16 tests (the chained one failed on a comma in a label)
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; OUTR=$RES/sub16_tests; Z=21
RUNS="sub16 prod (pca warm 100 fr):$D/$RES/invivo_prod/sub16_sl${Z}_s0,tofts8 prod:$D/$RES/invivo_prod/tofts8_sl${Z}_s0"
for T in nowarm nocap toftsinit phitv0.03 phitv0.3 w0_10 ortho; do RUNS="$RUNS,sub16 $T:$D/$OUTR/$T"; done
$P sub16_atoms_diag.py --slice $Z --runs "$RUNS" --tag _tests; echo "ATOMS exit $?"
