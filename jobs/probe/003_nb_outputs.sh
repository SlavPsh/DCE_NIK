#!/bin/bash
# inspect the executed grasp v2 notebook on helios: outputs of the section 10 cells added 2026-09-10 and the big picture cell
cd /net/beegfs/users/P101440/DCE_NIK || exit 1
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
NB=results/xcat_physical_nomotion_nik_vs_grasp/NIK_vs_GRASPv2_combined_report.ipynb
ls -la $NB $NB.bak_* 2>/dev/null
micromamba run -n torch29 python - "$NB" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); cells = d["cells"]; print(len(cells), "cells")
i0 = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith("### rank 8 basis"))
for i in range(i0, len(cells)):
    c = cells[i]; src = "".join(c["source"]).strip().split("\n")[0][:70]
    if c["cell_type"] != "code": print(f"\n[{i}] md  {src}"); continue
    outs = c.get("outputs", []); kinds = [o.get("output_type") + ("/" + ",".join(k for k in o.get("data", {}) if k != "text/plain") if o.get("data") else "") for o in outs]
    print(f"\n[{i}] code exec={c.get('execution_count')} {src!r}\n     outputs {len(outs)}: {kinds[:12]}")
    for o in outs[:3]:
        t = o.get("text") or (o.get("data", {}).get("text/plain")) or (o.get("ename", "") + ": " + o.get("evalue", ""))
        if t: print("     | " + ("".join(t) if isinstance(t, list) else str(t)).strip().replace("\n", "\n     | ")[:600])
PY
