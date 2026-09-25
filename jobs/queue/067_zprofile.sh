#!/bin/bash
#SBATCH -J gv2zprof
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:40:00
# per-slice intensity profile of the prep (straight after the kz ifft) and of all 7 recon tags, split into the smooth z
# trend (coil falloff, slab profile, anatomy = shading) and the slice-to-slice residual (= what actually stripes a
# coronal cut). answers whether the banding is already in the data before reconstruction.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; G=/net/beegfs/users/P101440/grasp_v2
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
export OUT=/net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun FIGDIR=$D/results/dce_rerun_gv2
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
cd $G && git log --oneline -1
for a in 1 2 3; do micromamba run -n torch29 python -u gv2_zprofile.py && break; echo "attempt $a failed"; sleep 30; done
