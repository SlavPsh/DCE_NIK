#!/bin/bash
# retrieval only: meas_p8_dce.dat existence + twix header facts (views, partitions, coils, TA, TR, flip, FOV), envs, and what results_ref holds
cd /net/beegfs/users/P101440
ls -la dce_data/orig/ | awk '{print $5, $9}'
echo "== envs"; micromamba env list 2>/dev/null; ls /appdata/users/P101440/micromamba/envs/ 2>/dev/null
echo "== results_ref"; ls grasp_pro_py/results_ref | head -30; ls grasp_pro_py/ | grep -i "results_ref\|results_spoke"
echo "== header p8 vs p3"
for F in meas_p8_dce.dat meas_p3_dce.dat; do echo "-- $F"; micromamba run -n torch29 python - <<PY
import re, struct
f = open("/net/beegfs/users/P101440/dce_data/orig/$F", "rb"); head = f.read(10240); n = struct.unpack("<I", head[4:8])[0]
print("measurements", n)
f.seek(0); txt = f.read(80_000_000).decode("latin-1")
for key in ("lRadialViews", "lPartitions", "dSliceResolution", "lImagesPerSlab", "alTR\[0\]", "adFlipAngleDegree\[0\]", "alTE\[0\]", "lBaseResolution", "dReadoutFOV", "dPhaseFOV", "dThickness", "lTotalScanTimeSec", "tPatientName", "tSequenceFileName", "lRepetitions", "lContrasts", "lAverages"):
    m = re.findall(r'%s\s*=\s*("?[^\n"]*"?)' % key, txt)
    print(key, sorted(set(m))[:4])
PY
done
