#!/bin/bash
#SBATCH -J claimfig
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:40:00
# images + curves behind the current claims: (a) regularization variants at 12k vs base and grasp; (b) coil-mode variants vs grasp
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=4
R=/net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak; AT=$R/amp_track; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; OC=/net/beegfs/users/P101440/DCE_NIK/results/realdata_nik_vs_cs_figures
P="micromamba run -n torch29 python -u"
$P compare_runs_fig.py --slice 21 --out $R/figures/claims_regularization_sl21.png --title "slice 21, k80, tofts8 unit-rms atoms, 12k steps: regularization variants vs base and GRASP (image at 90 s)" --items \
"base wd3e-3:$AT/rms1_wd3e-3_sl21/snap_slice_21_step012000.npy,wd 1e-2:$AT/rms1_wd1e-2_sl21/snap_slice_21_step012000.npy,coef TV 0.3:$AT/ctv0.3_sl21/snap_slice_21_step012000.npy,support 3 (v1):$AT/sup3_sl21/snap_slice_21_step012000.npy,pisco 1 (v1):$AT/pisco1_sl21/snap_slice_21_step012000.npy,GRASP:$GV/gv2_slice21_n12_k80.npy"; echo "FIG1 exit $?"
$P compare_runs_fig.py --slice 21 --out $R/figures/claims_coilmode_sl21.png --title "slice 21, k80, same trainer and protocol (10k, no restore): coil parameterization vs GRASP (image at 90 s)" --items \
"tofts8 input-coil:$R/invivo_k80_rms1/tofts8_sl21_s0/nik_slice_21_cplx.npy,tofts8 output-coil:$R/invivo_k80_oc/tofts8_sl21_s0/nik_slice_21_cplx.npy,sub16 input-coil:$R/invivo_k80_rms1/sub16_sl21_s0/nik_slice_21_cplx.npy,sub16 output-coil (old, outcoil_real):$OC/outcoil_subspace_output_slice21.npy,GRASP:$GV/gv2_slice21_n12_k80.npy"; echo "FIG2 exit $?"
