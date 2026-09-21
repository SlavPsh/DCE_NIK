#!/bin/bash
# which recon4d.nii.gz exist for the DCE_Rerun grasp v2 tags: size, date, dims, voxel, TR, and the per-slice npy still kept
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH PYTHONWARNINGS=ignore
R=/net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun
ls -la $R/*/recon4d.nii.gz 2>&1
echo "== 3d summaries and per-slice files"
for d in $R/*/; do printf "%-28s nii %s  slice_npy %s  total %s\n" "$(basename $d)" "$(ls $d/recon4d.nii.gz 2>/dev/null | wc -l)" "$(ls $d/slice_*.npy 2>/dev/null | wc -l)" "$(du -sh $d 2>/dev/null | cut -f1)"; done
echo "== headers"
timeout 50 micromamba run -n torch29 python -u - <<'PY'
import glob, os, nibabel as nib, numpy as np
for p in sorted(glob.glob("/net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun/*/recon4d.nii.gz")):
    im = nib.load(p); h = im.header; z = np.diag(im.affine)[:3]
    print(f"{os.path.basename(os.path.dirname(p)):10s} {im.shape} voxel {z[0]:.3f}x{z[1]:.3f}x{z[2]:.2f} mm  TR {float(h['pixdim'][4]):.2f} s  {os.path.getsize(p)/1e6:.0f} MB")
PY
