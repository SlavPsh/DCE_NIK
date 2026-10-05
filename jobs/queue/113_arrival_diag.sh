#!/bin/bash
#SBATCH -J arrival
#SBATCH -p gpu
#SBATCH --gres=gpu:1g.12gb:1
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH -t 2:00:00
# artifacts at contrast arrival (user observation 2026-10-05): time-resolved artifact metrics per arm, per-atom coefficient images with
# effective spoke counts, and the per-spoke k-space residual vs time of the tofts8 model. p3 slice 21 (production arms) and p14 slice 24
# (sigma-5 arms). no retraining.
set -uo pipefail
D=/net/beegfs/users/P101440/DCE_NIK; cd $D
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
P="micromamba run -n torch29 python -u"; RES=results/tofts_vs_patlak
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"; nvidia-smi -L || true
Z=21; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
DCE_DS=p3 $P arrival_artifact_diag.py --slice $Z --tag _prod --model "tofts8 in-coil+prior:$D/$RES/invivo_prod/tofts8_sl${Z}_s0" --items "tofts8 in-coil+prior:$D/$RES/invivo_prod/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$RES/invivo_prod_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$RES/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3)+prior:$D/$RES/invivo_prod_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$RES/invivo_prod/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"; echo "P3 exit $?"
DCE_DS=p3 $P arrival_artifact_diag.py --slice $Z --tag _prod_patlak --model "patlak+prior:$D/$RES/invivo_prod/patlak_sl${Z}_s0" --items "patlak+prior:$D/$RES/invivo_prod/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy"; echo "P3 patlak exit $?"
Z=24; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14; OUTR=$RES/p14
DCE_DS=p14 $P arrival_artifact_diag.py --slice $Z --tag _ks5 --model "tofts8 in-coil+prior:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s0" --items "tofts8 in-coil+prior:$D/$OUTR/invivo_prod_ks5/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,tofts8 out-coil+prior:$D/$OUTR/invivo_prod_ks5_oc/tofts8_sl${Z}_s0/nik_slice_${Z}_cplx.npy,patlak+prior:$D/$OUTR/invivo_prod_ks5/patlak_sl${Z}_s0/nik_slice_${Z}_cplx.npy,sub16 (wd 3e-3)+prior:$D/$OUTR/invivo_prod_ks5_sub16wd/sub16_sl${Z}_s0/nik_slice_${Z}_cplx.npy,NIK-free:$D/$OUTR/invivo_prod_ks5/free_sl${Z}_s0/nik_slice_${Z}_cplx.npy,GRASP:$GV/gv2_slice${Z}_n12_k80.npy,GRASP-Pro:$GP/cs_slice${Z}_f80match.npy"; echo "P14 exit $?"
