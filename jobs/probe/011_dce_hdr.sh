#!/bin/bash
# dce sequence parameters (TR, TE, flip) from the twix header text of meas_p3_dce.dat, first 60 mb only
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
timeout 50 micromamba run -n torch29 python -u - <<'PY'
import re
txt = open("/net/beegfs/users/P101440/dce_data/orig/meas_p3_dce.dat", "rb").read(60_000_000).decode("latin-1")
for key in ("adFlipAngleDegree[0]", "alTR[0]", "alTE[0]", "lTotalScanTimeSec", "lRadialViews", "sKSpace.lBaseResolution", "flNominalB0", "lContrasts", "tSequenceFileName", "ulVersion"):
    m = re.findall(re.escape(key) + r"\s*=\s*([-0-9.\"a-zA-Z_%/\]+)", txt); print(f"{key:28s} {m[:3]}")
PY
