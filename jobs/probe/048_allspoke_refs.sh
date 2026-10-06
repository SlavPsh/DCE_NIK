#!/bin/bash
# retrieval only: which all-spoke cs references exist (grasp v2 n12 without k80, grasp-pro f100 / cs_img frames) for p3 and p14
cd /net/beegfs/users/P101440; ls -la --time-style=+%m-%d grasp_v2/results_grasp_v2/gv2_slice{18,19,21}_n12*.npy grasp_v2/results_grasp_v2_p14/gv2_slice*_n12*.npy grasp_pro_py/results_spoke_cs/cs_slice21_f100*.npy grasp_pro_py/results_spoke_cs_p14/*.npy 2>&1 | awk '{print $5, $6, $7}'
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
micromamba run -n torch29 python -c "
import numpy as np
for p in ['grasp_pro_py/results_ref/slice_21.npz','grasp_pro_py/results_ref_p14/slice_24.npz']:
    z=np.load(p); print(p, 'cs_img', z['cs_img'].shape, [k for k in z.files])"
ls DCE_NIK/spoke_masks/
