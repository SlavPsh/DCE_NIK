#!/bin/bash
# the first parametric aif had a kink at the recirculation onset (gamma exponent at its bound): cancel the param run (task 0 of 3469930), refit with
# smooth-onset bounds, rebuild the param basis, resubmit the param task alone (no chained eval), and re-chain the eval after both arrays
cd /net/beegfs/users/P101440/DCE_NIK; export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH; P="micromamba run -n torch29 python -u"
scancel 3469930_0 && echo "cancelled 3469930_0"; sleep 5
rm -f aif_param_slice21.npz results/tofts_vs_patlak/basis_sl21_r8_param.npz results/tofts_vs_patlak/basis_sl21_r8_param_rms1.npz; rm -rf results/tofts_vs_patlak/arrival_fix/param
DCE_DS=p3 $P aif_param_fit.py --slice 21 2>&1 | grep -E "fit rms|params|Traceback|Error" | cut -c1-300
$P nik_tofts_basis.py --aif aif_param_slice21.npz --out results/tofts_vs_patlak/basis_sl21_r8_param.npz --ranks 8 > results/tofts_vs_patlak/basis_sl21_r8_param.log 2>&1; $P basis_rms1.py results/tofts_vs_patlak/basis_sl21_r8_param.npz
grep -E "^ *8 " results/tofts_vs_patlak/basis_sl21_r8_param.log
J=$(sbatch --parsable --array=0 --export=ALL,STAGE=train,NOCHAIN=1 -J arrfix_param jobs/queue/114_arrival_fixes.sh); echo "param resubmitted $J"
scancel 3470094 && echo "cancelled eval 3470094"
sbatch --dependency=afterany:3469930:$J --export=ALL,STAGE=eval --array=0 -J arrfix_eval --partition=gpu --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 4:00:00 --output=jobs/log/114_arrival_fixes_eval_%j.out --error=jobs/log/114_arrival_fixes_eval_%j.out jobs/queue/114_arrival_fixes.sh
git add results/realdata_nik_vs_cs_figures/figures/aif_param_slice21.png && echo "fig staged"
