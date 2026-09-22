#!/bin/bash
#SBATCH -J gradchk
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 1:00:00
# gradient check on the production models (tofts8 in-coil, tofts8 out-coil, sub16, free): no complex tensors in the data-loss graph, kink ops, autograd vs finite differences in float64
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=8
R=/net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak
micromamba run -n torch29 python -u gradient_check.py --slice 21 --batch 4096 --runs "tofts8 in-coil:$R/invivo_prod/tofts8_sl21_s0,tofts8 out-coil:$R/invivo_prod_oc/tofts8_sl21_s0,sub16:$R/invivo_prod/sub16_sl21_s0,free:$R/invivo_prod/free_sl21_s0"
echo "GRADCHK exit $?"
