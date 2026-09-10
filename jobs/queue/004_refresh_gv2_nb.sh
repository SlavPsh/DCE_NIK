#!/bin/bash
#SBATCH -J refreshgv2
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 2:00:00
# re-execute the grasp v2 report notebook (section 10 rank 8 cells, inline figures, big picture); the agent commits the ipynb
set -uo pipefail
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
micromamba run -n torch29 python -u refresh_gv2_nb.py
echo "REFRESH exit $?"
