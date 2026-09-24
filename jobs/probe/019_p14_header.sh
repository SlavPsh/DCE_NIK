#!/bin/bash
# retrieval only: meas_topqmri_p14.dat existence, size, twix header facts vs p3
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
cd /net/beegfs/users/P101440
ls -la dce_data/orig/ | awk '{print $5, $9}'
for F in meas_topqmri_p14.dat meas_p3_dce.dat; do echo "-- $F"; micromamba run -n torch29 python - <<PY
import re, struct, os
p = "/net/beegfs/users/P101440/dce_data/orig/$F"
if not os.path.exists(p): print("MISSING", p); raise SystemExit
f = open(p, "rb"); head = f.read(10240); n = struct.unpack("<I", head[4:8])[0]; print("size", os.path.getsize(p), "measurements", n)
f.seek(0); txt = f.read(80_000_000).decode("latin-1")
for key in ("lRadialViews", "lPartitions", "dSliceResolution", "lImagesPerSlab", "alTR\\[0\\]", "adFlipAngleDegree\\[0\\]", "alTE\\[0\\]", "lBaseResolution", "dReadoutFOV", "dThickness", "lTotalScanTimeSec", "tSequenceFileName", "tProtocolName", "lRepetitions", "lContrasts", "lAverages", "sKSpace.ucTrajectory", "ucTrajectory"):
    m = re.findall(r'%s\s*=\s*("?[^\n"]*"?)' % key, txt)
    print(key.replace("\\\\", ""), sorted(set(x.strip() for x in m))[:6])
PY
done
