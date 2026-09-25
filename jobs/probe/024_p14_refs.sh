#!/bin/bash
# confirm the p14 grasp references and the first nik runs' progress
ls -la --time-style=+%H:%M /net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14/ /net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs_p14/ 2>/dev/null | awk '{print $5, $6, $7}'
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
micromamba run -n torch29 python -c "import numpy as np; [print(f, np.load(f'/net/beegfs/users/P101440/grasp_v2/results_grasp_v2_p14/gv2_slice{z}_n12_k80.npy', mmap_mode='r').shape) for z in (21,24,27) for f in [f'gv2 {z}']]"
cd /net/beegfs/users/P101440/DCE_NIK; for d in results/tofts_vs_patlak/p14/invivo_prod/tofts8_sl21_s0 results/tofts_vs_patlak/p14/invivo_prod/tofts8_sl24_s0; do echo "== $d"; grep "step " $d/train.log 2>/dev/null | tail -2; done; squeue -u $USER -h -o "%i %j %T %M" | head -6
