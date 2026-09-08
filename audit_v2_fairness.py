"""independent check of the grasp v2 vs NIK comparison. three questions:
 1 same input data?
 2 is the curve metric biased by unmatched temporal sampling? (grasp 43 frames vs NIK 344)
 3 one setting selected on phantom, or best-of cherry pick?
"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, glob, os
import numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK")
import xph_pipeline as P, xph_common as X

SW = f"{P.OUT}/v2_sweep"; A = f"{P.OUT}/arrays"
d = P.data(); tq = d["times"]; body = d["labels"] > 0
Rz = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq); F = len(tq)

print("=== 1. INPUT DATA ===")
z = np.load(f"{A}/xph_slice_cache.npz"); kdata = z["kdata"]
print(f"  phantom kdata {kdata.shape} (C,F,angles,RO); TRAIN_ANG={P.TRAIN_ANG} VAL={P.VAL_ANG} TEST={P.TEST_ANG}")
print(f"  grasp v2 sweep uses tr=TRAIN_ANG only -> {F}*{len(P.TRAIN_ANG)} = {F*len(P.TRAIN_ANG)} spokes")
print(f"  NIK trains on TRAIN_ANG only          -> {F*len(P.TRAIN_ANG)} spokes   SAME")
print("  grouping G only REBINS those spokes, never adds any")

print("\n=== 2. TEMPORAL SAMPLING BIAS (the suspicious part) ===")
print("  grasp v2 at 40 spf outputs 43 frames; NIK outputs 344.")
print("  curve NRMSE on a 43-frame curve is inherently DENOISED vs a 344-frame one.")
print("  -> re-score NIK with its curve window-averaged to the SAME frame count.\n")
niks = {n: np.load(f"{A}/nik_fine_{n}.npy") for n in ("sub16", "free") if os.path.exists(f"{A}/nik_fine_{n}.npy")}

def curve_native(v, t, roi):
    return np.interp(tq, t, np.array([v[..., i][Rz[roi]].mean() for i in range(v.shape[-1])]))

def curve_binned(v, G, roi):
    nG = v.shape[-1] // G
    vg = np.stack([v[:, :, g*G:(g+1)*G].mean(2) for g in range(nG)], -1)
    tG = np.array([tq[g*G:(g+1)*G].mean() for g in range(nG)])
    return np.interp(tq, tG, np.array([vg[..., i][Rz[roi]].mean() for i in range(nG)]))

G_SEL = 8   # 40 spokes/frame, the tuned operating point
rows = []
for nm, v in niks.items():
    for lab, c_fn in (("native 344fr", lambda r: curve_native(v, tq, r)),
                      (f"binned to {F//G_SEL}fr", lambda r: curve_binned(v, G_SEL, r))):
        r = dict(method=f"NIK-{nm} {lab}")
        for roi in ("aorta", "cortex", "medulla"):
            c = c_fn(roi); ct = Tr[Rz[roi]].mean(0)
            r[roi] = float(np.linalg.norm(c-ct)/(np.linalg.norm(ct)+1e-12))
        r["peak"] = float(c_fn("aorta").max()); rows.append(r)
for f in (f"{SW}/v2_G08_lam0.02.npy", f"{SW}/v2_G08.npy"):
    if not os.path.exists(f): continue
    v = np.load(f); nG = v.shape[-1]
    tG = np.array([tq[g*G_SEL:(g+1)*G_SEL].mean() for g in range(nG)])
    lam = "0.02" if "lam" in f else "0.25"
    r = dict(method=f"GRASP-v2 40spf lam{lam} ({nG}fr)")
    for roi in ("aorta", "cortex", "medulla"):
        c = np.interp(tq, tG, np.array([v[..., i][Rz[roi]].mean() for i in range(nG)])); ct = Tr[Rz[roi]].mean(0)
        r[roi] = float(np.linalg.norm(c-ct)/(np.linalg.norm(ct)+1e-12))
    r["peak"] = float(np.interp(tq, tG, np.array([v[..., i][Rz["aorta"]].mean() for i in range(nG)])).max())
    rows.append(r)
print(f"{'method':34} {'aorta':>8} {'cortex':>8} {'medulla':>8} {'peak':>8}")
for r in rows:
    print(f"{r['method']:34} {r['aorta']:>8.4f} {r['cortex']:>8.4f} {r['medulla']:>8.4f} {r['peak']:>8.4f}")
print(f"  truth peak {float(Tr[Rz['aorta']].mean(0).max()):.4f}")

print("\n=== 3. SELECTION ===")
fr = json.load(open(f"{SW}/frontier.json")); g = [x for x in fr if x["method"] == "GRASP-v2"]
print("  grasp v2 lam0.25 best-per-metric comes from DIFFERENT spf:")
print(f"    haarpsi best {max(g,key=lambda x:x['haarpsi'])['spf']} spf | aortaC best {min(g,key=lambda x:x['c_aorta'])['spf']} spf")
print("  -> reporting both is a best-of cherry pick. ONE setting must be fixed on the phantom.")
sel = min(g, key=lambda r: (1-r["haarpsi"])**2 + r["c_aorta"]**2)
print(f"  ideal-corner single pick (lam0.25): {sel['spf']} spf = {sel['dt']:.2f} s/frame")
print(f"  -> in vivo equivalent NLINE = {round(sel['dt']/(375.0/1710)):d} (matched seconds/frame, NOT re-selected in vivo)")
print("AUDIT_DONE")
