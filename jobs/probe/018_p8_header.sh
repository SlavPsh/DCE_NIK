#!/bin/bash
# retrieval only (017 lacked the micromamba path): p8 vs p3 twix header facts, twixtools availability in torch29
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
cd /net/beegfs/users/P101440
echo "== twixtools in torch29"; micromamba run -n torch29 python -c "import twixtools, sys; print('twixtools', getattr(twixtools, '__version__', 'ok'))" 2>&1 | tail -1
for F in meas_p8_dce.dat meas_p3_dce.dat; do echo "-- $F"; micromamba run -n torch29 python - <<PY
import re, struct
f = open("/net/beegfs/users/P101440/dce_data/orig/$F", "rb"); head = f.read(10240); n = struct.unpack("<I", head[4:8])[0]
print("measurements", n)
f.seek(0); txt = f.read(80_000_000).decode("latin-1")
for key in ("lRadialViews", "lPartitions", "dSliceResolution", "lImagesPerSlab", "alTR\\[0\\]", "adFlipAngleDegree\\[0\\]", "alTE\\[0\\]", "lBaseResolution", "dReadoutFOV", "dPhaseFOV", "dThickness", "lTotalScanTimeSec", "tSequenceFileName", "lRepetitions", "tPatientName", "FrameOfReference", "tPatientPosition"):
    m = re.findall(r'%s\s*=\s*("?[^\n"]*"?)' % key, txt)
    print(key.replace("\\\\", ""), sorted(set(x.strip() for x in m))[:4])
PY
done
