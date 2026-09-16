#!/bin/bash
#SBATCH -J gv2rerun
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -t 6:00:00
# grasp v2 (mcnufft + temporal tv + nlcg) on DCE_Rerun meas_MID3000007_FID100985, full slab, 21 spokes/frame (3.6 s),
# lambdas 0.02 / 0.08 / 0.25 of max|E'y| plus the plain nufft. stage prep: `python -m dce prep` = global coil compression
# to 8, kz block placed by mdh CenterPar on the 64-line grid (dSliceResolution 0.5), POCS partial Fourier on the missing
# low-kz lines, z-ifft, slab-oversampling crop to lImagesPerSlab, per-slice grog b1. then chains an array (one task per
# slice, all lambdas, shared E and x0) and a post stage (frame times, 4d nifti per lambda, ortho views, curves).
# output volumes: /net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun/<tag>/recon4d.nii.gz; figures + README in
# DCE_NIK/results/dce_rerun_gv2 (flow back through git)
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; G=/net/beegfs/users/P101440/grasp_v2; P=/net/beegfs/users/P101440/grasp_pro_py
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONDONTWRITEBYTECODE=1
export DAT=/net/beegfs/users/P101440/dce_data/orig/meas_MID3000007_FID100985_DCE_Rerun_1.dat
export OUT=/net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun NLINE=21 LAMS=0.02,0.08,0.25 FIGDIR=$D/results/dce_rerun_gv2
STAGE=${STAGE:-prep}
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:-} stage $STAGE cpus ${SLURM_CPUS_PER_TASK:-?}"
case $STAGE in
prep)
  cd $P && git log --oneline -1; cd $G && git log --oneline -1
  PREPENV=""
  for e in torch29 anon; do micromamba run -n $e python -c "import twixtools" 2>/dev/null && { PREPENV=$e; break; }; done
  [ -z "$PREPENV" ] && { echo "NO env with twixtools"; micromamba env list; exit 1; }
  echo "prep env $PREPENV"
  ls -la $DAT || exit 1
  if [ ! -f $OUT/prep/shared.npz ]; then
    cd $P; micromamba run -n $PREPENV python -u -m dce prep --file $DAT --out $OUT/prep --ncc 8 --cut 15 || { echo "PREP FAILED"; exit 1; }
  else echo "prep cached"; fi
  cd $G; micromamba run -n torch29 python -u gv2_rerun.py geom || exit 1
  NZ=$(micromamba run -n torch29 python -c "import numpy as np; print(int(np.load('$OUT/prep/shared.npz')['nzz']))")
  echo "slices $NZ"; du -sh $OUT/prep
  A=$(sbatch --parsable --dependency=afterok:$SLURM_JOB_ID --export=ALL,STAGE=recon --array=0-$((NZ-1))%12 -J gv2rr -p defq -c 8 --mem 16G -t 4:00:00 \
      --output=$D/jobs/log/043_gv2_rerun_recon_%A_%a.out --error=$D/jobs/log/043_gv2_rerun_recon_%A_%a.out $D/jobs/queue/043_gv2_rerun.sh) || exit 1
  B=$(sbatch --parsable --dependency=afterok:$A --export=ALL,STAGE=post -J gv2post -p defq -c 4 --mem 32G -t 2:00:00 \
      --output=$D/jobs/log/043_gv2_rerun_post_%j.out --error=$D/jobs/log/043_gv2_rerun_post_%j.out $D/jobs/queue/043_gv2_rerun.sh) || exit 1
  echo "chained recon array $A (0-$((NZ-1)), 12 concurrent) -> post $B"
  echo "PREP_DONE $(date '+%F %T')"
  ;;
recon)
  cd $G; micromamba run -n torch29 python -u gv2_rerun.py recon --slice $SLURM_ARRAY_TASK_ID
  ;;
post)
  cd $G; micromamba run -n torch29 python -u gv2_rerun.py post && ls -la $FIGDIR
  ;;
esac
