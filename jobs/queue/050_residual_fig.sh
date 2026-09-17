#!/bin/bash
#SBATCH -J residfig
#SBATCH -p defq
#SBATCH -c 4
#SBATCH --mem 24G
#SBATCH -t 0:30:00
# residual maps vs the all-spoke anatomy at 90 s: streak texture vs anatomy for grasp, grasp-pro and the tofts8 candidates
cd /net/beegfs/users/P101440/DCE_NIK
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH OMP_NUM_THREADS=4
R=/net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak; GV=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2; GP=/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs
micromamba run -n torch29 python -u residual_fig.py --slice 21 --out $R/figures/residual_sl21.png --items \
"GRASP:$GV/gv2_slice21_n12_k80.npy,GRASP-Pro:$GP/cs_slice21_f80match.npy,NIK-tofts8 old (3k restore):$R/invivo_k80/tofts8_sl21_s0/nik_slice_21_cplx.npy,NIK-tofts8 10k wd3e-3:$R/invivo_k80_rms1/tofts8_sl21_s0/nik_slice_21_cplx.npy,NIK-tofts8 wd1e-2 step 8k:$R/amp_track/rms1_wd1e-2_sl21/snap_slice_21_step008000.npy,NIK-free:/net/beegfs/users/P101440/DCE_NIK/results_sl21_k80/nik_slice_21.npy"
echo "RESID exit $?"
