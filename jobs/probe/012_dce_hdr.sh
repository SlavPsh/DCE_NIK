#!/bin/bash
# dce sequence parameters from the twix header text (simple patterns, no character classes with backslashes)
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
timeout 50 micromamba run -n torch29 python -u - <<'PY'
import re
txt = open("/net/beegfs/users/P101440/dce_data/orig/meas_p3_dce.dat", "rb").read(60_000_000).decode("latin-1")
for key in ("adFlipAngleDegree[0]", "alTR[0]", "alTE[0]", "lTotalScanTimeSec", "lRadialViews", "flNominalB0", "tSequenceFileName"):
    i = txt.find(key); print(f"{key:24s} {txt[i:i+60].splitlines()[0] if i >= 0 else 'not found'}")
PY
