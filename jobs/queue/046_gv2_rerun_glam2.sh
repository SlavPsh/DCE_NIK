#!/bin/bash
#SBATCH -J gv2glam2
#SBATCH -p defq
#SBATCH -c 2
#SBATCH --mem 4G
#SBATCH -t 0:20:00
# step 2 of the z-banding check: grasp v2 lam 0.25 with ONE global lambda for every slice (0.25 x slab median of
# max|E'y| = 7.58e-5) instead of the per-slice maximum (which spans 2.9e-5 to 11.5e-5, 17 of 53 neighbour jumps > 15%).
# tags glam0.02 / glam0.08 added (glam0.25 cached); post regenerates all figures with the global rows
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK
export GLAMS=0.02,0.08,0.25 X0REF=7.58e-5   # via --export=ALL: sbatch splits --export on commas
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
cd /net/beegfs/users/P101440/grasp_v2 && git log --oneline -1 && grep -q GLAMS gv2_rerun.py || { echo "driver without GLAMS"; exit 1; }
A=$(sbatch --parsable --export=ALL,STAGE=recon,PYTHONWARNINGS=ignore --array=0-53%24 -J gv2rr4 -p defq -c 8 --mem 16G -t 4:00:00 \
    --output=$D/jobs/log/046_gv2_rerun_glam2_recon_%A_%a.out --error=$D/jobs/log/046_gv2_rerun_glam2_recon_%A_%a.out $D/jobs/queue/043_gv2_rerun.sh) || exit 1
B=$(sbatch --parsable --dependency=afterok:$A --export=ALL,STAGE=post,PYTHONWARNINGS=ignore -J gv2post4 -p defq -c 4 --mem 32G -t 2:00:00 \
    --output=$D/jobs/log/046_gv2_rerun_glam2_post_%j.out --error=$D/jobs/log/046_gv2_rerun_glam2_post_%j.out $D/jobs/queue/043_gv2_rerun.sh) || exit 1
echo "chained recon array $A -> post $B"; echo "GLAM_DONE"
