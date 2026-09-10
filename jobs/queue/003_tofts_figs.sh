#!/bin/bash
#SBATCH -J tfigs
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:40:00
# in vivo figures for every arm present (patlak, tofts, tofts8) on slices 18/19/21 -> wandb group tofts_figs
set -uo pipefail
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
micromamba run -n torch29 python -u tofts_figs_wandb.py --slices 18,19,21 --suffix _r8
echo "FIGS exit $?"
