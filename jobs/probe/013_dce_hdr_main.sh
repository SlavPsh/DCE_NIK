#!/bin/bash
# dce scan parameters from the LAST measurement header of the twix file (adjustment scans come first)
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
timeout 50 micromamba run -n torch29 python -u - <<'PY'
import re, struct
f = open("/net/beegfs/users/P101440/dce_data/orig/meas_p3_dce.dat", "rb"); head = f.read(10240); n = struct.unpack("<I", head[4:8])[0]
ent = [(struct.unpack("<I", head[8+152*k:12+152*k])[0], struct.unpack("<Q", head[16+152*k:24+152*k])[0], struct.unpack("<Q", head[24+152*k:32+152*k])[0], head[96+152*k:160+152*k].split(b"\x00")[0].decode("latin-1", "replace")) for k in range(n)]
print("measurements (id, offset, length, protocol):"); [print("  ", e) for e in ent]
f.seek(ent[-1][1]); txt = f.read(8_000_000).decode("latin-1")
for key in ("tSequenceFileName", "adFlipAngleDegree[0]", "alTR[0]", "alTE[0]", "lRadialViews", "lTotalScanTimeSec", "tProtocolName"):
    hits = [txt[m.start():m.start()+70].splitlines()[0] for m in re.finditer(re.escape(key), txt)]; print(f"{key:22s} {len(hits)} hits: {hits[:2]} ... {hits[-1:] if len(hits) > 2 else ''}")
PY
