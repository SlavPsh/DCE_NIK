#!/bin/bash
# per-tissue pk / T1 lookup table of the xcat simulator, the sim hdf5 layout, and the labels present in slice ZI
cd /net/beegfs/users/P101440/DCE_NIK || exit 1
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH XPH_SIM=nomotion
M=$(find /net/beegfs/users/P101440 -maxdepth 4 -name 'XCAT_to_MR_DCE.m' 2>/dev/null | head -1); echo "== $M"
[ -n "$M" ] && sed -n '80,175p' "$M"
timeout 40 micromamba run -n torch29 python -u - <<'PY'
import h5py, numpy as np, xph_common as X, xph_pipeline as P
print("== SIM", X.SIM); f = h5py.File(X.SIM, "r"); f.visit(lambda n: print("  ", n, f[n].shape if hasattr(f[n], "shape") else "") if n.count("/") < 3 else None)
for k, v in f["results"].attrs.items(): print("  attr", k, v)
d = P.data(); lab = d["labels"]; u, c = np.unique(lab, return_counts=True); print("== labels in slice", P.ZI, dict(zip(u.tolist(), c.tolist())))
PY
