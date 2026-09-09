#!/bin/bash
#SBATCH -J xphv2agg
#SBATCH -p gpu
#SBATCH --gres gpu:1g.12gb:1
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 4:00:00
#SBATCH -o /net/beegfs/users/P101440/DCE_NIK/xphv2agg_%j.log
cd /net/beegfs/users/P101440/DCE_NIK
P=${PY:-/net/beegfs/users/P101440/micromamba/envs/torch29/bin/python}
export GRASP_LABEL="GRASP-v2" TAG=_gv2
run(){ echo; echo "=== $* ==="; "$@" || echo "FAILED: $*"; }
GRASP_NPZ=grasp_v2_recon.npz run $P -u xph_aggregate.py
GRASP_NPZ=grasp_v2_recon.npz run $P -u l3_rebaseline.py
GRASP_NPZ=grasp_v2_recon.npz run $P -u render_compare.py
echo XPH_V2_AGG_DONE
