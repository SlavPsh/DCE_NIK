"""matched-IMAGE-QUALITY comparison of contrast curves, phantom.

the point: a contrast-curve comparison is confounded if the two recons differ in image quality.
grasp v2's first phantom run sat at 5 spokes/frame, the worst point on its own frontier
(haarpsi 0.44), so its curves could not be attributed to its temporal model.

here grasp v2 is swept over BOTH knobs that govern image quality and bolus damping:
  spokes/frame (25/40/60) and lam (0.02 to 0.50 x max|x0|; 0.25 is the published value).
LOWER lam = weaker temporal TV = less bolus flattening but more noise, so this is the setting most
able to recover the aorta peak. we then compare curves ONLY among configs whose image quality
reaches NIK's best, which is the fair comparison.
"""
import warnings; warnings.filterwarnings("ignore")
import json, glob, os, sys
import numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK")
import xph_pipeline as P

SW = f"{P.OUT}/v2_sweep"
FR = json.load(open(f"{SW}/frontier.json"))
NIK = {m: max(x["haarpsi"] for x in FR if x["method"] == m) for m in ("NIK-sub16", "NIK-free")}
NIKBEST = max(NIK.values()); NIKBEST_M = max(NIK, key=NIK.get)
nik_rows = {m: sorted([x for x in FR if x["method"] == m], key=lambda z: -z["haarpsi"])[0] for m in NIK}

rows = []
for f in sorted(glob.glob(f"{SW}/v2_G*.json")):
    r = json.load(open(f))
    r.setdefault("lam_frac", 0.25)
    rows.append(r)
rows.sort(key=lambda r: (r["spokes_per_frame"], r["lam_frac"]))

TRUTH_PK = rows[0].get("aorta_pk_truth", float("nan"))
print("phantom, grasp v2 swept over spokes/frame AND lam. reference = xcat truth.")
print(f"NIK best image quality: {NIKBEST_M} haarpsi {NIKBEST:.4f}   (truth aorta peak {TRUTH_PK:.4f})\n")
print(f"{'spf':>4} {'lam':>6} {'frames':>7} {'HaarPSI':>8} {'>=NIK':>6} {'aortaC':>7} {'aortaPk':>8} {'pk err %':>9}")
for r in rows:
    if r["spokes_per_frame"] < 25: continue
    ok = "YES" if r["haarpsi"] >= NIKBEST else ""
    pe = 100 * (r["aorta_pk"] - TRUTH_PK) / TRUTH_PK
    print(f"{r['spokes_per_frame']:>4} {r['lam_frac']:>6.2f} {r['n_frames']:>7} {r['haarpsi']:>8.4f} {ok:>6}"
          f" {r['c_aorta']:>7.4f} {r['aorta_pk']:>8.4f} {pe:>+9.1f}")

matched = [r for r in rows if r["haarpsi"] >= NIKBEST]
print(f"\n{len(matched)} grasp v2 configs reach NIK's best image quality ({NIKBEST:.4f})")
if matched:
    bp = max(matched, key=lambda r: r["aorta_pk"])
    bc = min(matched, key=lambda r: r["c_aorta"])
    print(f"  best aorta PEAK among them : {bp['spokes_per_frame']} spf lam {bp['lam_frac']:.2f} -> "
          f"peak {bp['aorta_pk']:.4f} ({100*(bp['aorta_pk']-TRUTH_PK)/TRUTH_PK:+.1f}% vs truth), haarpsi {bp['haarpsi']:.4f}")
    print(f"  best aorta CURVE among them: {bc['spokes_per_frame']} spf lam {bc['lam_frac']:.2f} -> "
          f"nrmse {bc['c_aorta']:.4f}, haarpsi {bc['haarpsi']:.4f}")
    print("\n  vs NIK at ITS best image quality:")
    for m, r in nik_rows.items():
        print(f"    {m:10} haarpsi {r['haarpsi']:.4f}  aortaC {r['c_aorta']:.4f}  peak {r['aorta_pk']:.4f} "
              f"({100*(r['aorta_pk']-TRUTH_PK)/TRUTH_PK:+.1f}% vs truth)")
    print("\n  READ: compare curves ONLY on the rows marked >=NIK. if grasp v2's best peak there is still")
    print("        well under truth while NIK is close, the bolus damping is the temporal model, not image quality.")
json.dump(rows, open(f"{SW}/lam_sweep.json", "w"), indent=1)
print("\nLAM_ANALYSIS_DONE")
