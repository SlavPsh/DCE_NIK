#!/bin/bash
#SBATCH -J arrdiag2
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 2:00:00
# the 114 eval's arrival diagnostic rebuilt every model with the rank-8 measured-aif basis: wrong atoms for the param run (its temporal profiles,
# n_eff and residual were computed with the measured-aif atoms) and a state_dict mismatch for rank 5 / 6. rerun those three with the right basis.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak; OUTR=$RES/arrival_fix; Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2
BASE=$D/$RES/invivo_prod/tofts8_sl${Z}_s0; OC=$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0
for TR in "param 8 _param_rms1" "rank5 5 _rms1" "rank6 6 _rms1"; do set -- $TR; T=$1; R=$2; SFX=$3
  DCE_DS=p3 $P arrival_artifact_diag.py --slice $Z --tag _arrfix_$T --basis $RES/basis_sl${Z}_r${R}${SFX}.npz --model "tofts$R $T:$D/$OUTR/$T/tofts${R}_sl${Z}_s0" --items "tofts$R $T:$D/$OUTR/$T/tofts${R}_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 prod in-coil:$BASE/nik_slice_${Z}_cplx.npy,tofts8 prod out-coil:$OC/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy"; echo "DIAG $T exit $?"
done
