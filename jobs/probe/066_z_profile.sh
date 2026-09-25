#!/bin/bash
# is the slice-to-slice intensity difference already there after the kz ifft, before any recon?
# preview.npy is the prep's per-slice time-average (post kz-ifft, post grog, coil-combined), so its z profile answers it.
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH PYTHONWARNINGS=ignore
timeout 55 micromamba run -n torch29 python -u - <<'PY'
import numpy as np
R = "/net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun/prep"
P = np.load(f"{R}/preview.npy")                      # [bas, bas, nzz], post kz-ifft prep image
print("preview", P.shape)
body = P > np.percentile(P, 70)                      # same spirit as the recon's body mask
prof = np.array([np.median(P[:, :, z][body[:, :, z]]) for z in range(P.shape[2])])
p99 = np.array([np.percentile(P[:, :, z], 99.5) for z in range(P.shape[2])])
print("z  median(body)   p99.5      rel_to_median_slice")
for z in range(P.shape[2]):
    print(f"{z:2d}  {prof[z]:.4e}  {p99[z]:.4e}  {prof[z]/np.median(prof):.2f}")
print(f"\nmedian-of-body profile: min {prof.min():.3e} (z{prof.argmin()}), max {prof.max():.3e} (z{prof.argmax()}), ratio {prof.max()/prof.min():.2f}")
print(f"p99.5 profile:          min {p99.min():.3e} (z{p99.argmin()}), max {p99.max():.3e} (z{p99.argmax()}), ratio {p99.max()/p99.min():.2f}")
print("PROFILE " + ",".join(f"{v:.6e}" for v in prof))
# b1 is renormalised per slice (b1 /= max|b1|), so the recon's coil combine partly rescales each slice
for z in (2, 13, 27, 40, 52):
    d = np.load(f"{R}/slice_{z:02d}.npz"); b1 = d["b1"]; den = (np.abs(b1)**2).sum(-1)
    k = d["kdata_radial"]; dc = np.abs(k[k.shape[0]//2]).mean()
    print(f"slice {z:2d}: |k| at readout centre {dc:.4e}, den {den.min():.3f}-{den.max():.3f}, b1 max {np.abs(b1).max():.3f}")
PY
