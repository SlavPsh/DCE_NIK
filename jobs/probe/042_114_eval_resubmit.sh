#!/bin/bash
# the chained eval 3469933 was submitted with the buggy script copy: cancel it and resubmit the eval stage from the fixed file (afterany the array);
# regenerate the aif fit figure (deterministic refit, same npz) on this node
cd /net/beegfs/users/P101440/DCE_NIK; export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
scancel 3469933 && echo "cancelled 3469933"
sbatch --dependency=afterany:3469930 --export=ALL,STAGE=eval --array=0 -J arrfix_eval --partition=gpu --gres=gpu:1g.12gb:1 -c 4 --mem 48G -t 4:00:00 --output=jobs/log/114_arrival_fixes_eval_%j.out --error=jobs/log/114_arrival_fixes_eval_%j.out jobs/queue/114_arrival_fixes.sh
DCE_DS=p3 micromamba run -n torch29 python -u aif_param_fit.py --slice 21 2>&1 | grep -E "fit rms|params|Traceback|Error" | cut -c1-300
git add results/realdata_nik_vs_cs_figures/figures/aif_param_slice21.png && echo "fig staged"
