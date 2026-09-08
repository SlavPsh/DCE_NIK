"""ONE reference setting for grasp v2, fixed on the phantom, carried to in vivo unchanged.

setting: 25 spokes/frame, lam 0.25 (published). chosen on the phantom by the ideal-corner rule on
(haarpsi vs truth, aorta nrmse vs truth). 2.61 s/frame -> in vivo NLINE=12 (2.63 s/frame). NOT
re-selected in vivo. no best-of across settings anywhere.

in-vivo haarpsi vs the model-free nufft recon is EXCLUDED: the reference is itself a recon and the
metric is not informative. in vivo reports contrast curves only.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json
import numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK")
import xph_pipeline as P, xph_common as X
import consolidated as C

A = f"{P.OUT}/arrays"; SW = f"{P.OUT}/v2_sweep"
GV = "/scratch/rnga/vvpshenov/grasp_v2/results_grasp_v2"
B = "/scratch/rnga/vvpshenov/DCE_NIK"
G_SEL, LAM_SEL, NLINE_SEL = 5, 0.25, 12          # 25 spokes/frame phantom, NLINE 12 in vivo
out = {"setting": dict(phantom_spf=5*G_SEL, phantom_frames=344//G_SEL, lam=LAM_SEL,
                       s_per_frame=round(5*G_SEL*344/1720*0.1044*10, 2), invivo_NLINE=NLINE_SEL)}

# ---------------- phantom, vs truth ----------------
d = P.data(); tq = d["times"]; body = d["labels"] > 0
Rz = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq); F = len(tq)
def sc(v):
    n = v.shape[-1]
    Tw = Tr if n == F else np.stack([Tr[:, :, g*(F//n):(g+1)*(F//n)].mean(2) for g in range(n)], -1)
    return v * (np.sum(v[body]*Tw[body]) / (np.sum(v[body]**2) + 1e-12))
def ph_score(v, t, lab):
    r = {"method": lab, "frames": int(v.shape[-1])}
    for roi in ("aorta", "cortex", "medulla"):
        c = np.interp(tq, t, np.array([v[..., i][Rz[roi]].mean() for i in range(v.shape[-1])]))
        ct = Tr[Rz[roi]].mean(0); r[roi] = float(np.linalg.norm(c-ct)/np.linalg.norm(ct))
    ca = np.interp(tq, t, np.array([v[..., i][Rz["aorta"]].mean() for i in range(v.shape[-1])]))
    r["peak"] = float(ca.max()); r["peak_err_pct"] = 100*(ca.max()-float(Tr[Rz["aorta"]].mean(0).max()))/float(Tr[Rz["aorta"]].mean(0).max())
    return r
fr = json.load(open(f"{SW}/frontier.json"))
ph = []
for nm in ("sub16", "free"):
    p = f"{A}/nik_fine_{nm}.npy"
    if os.path.exists(p):
        r = ph_score(sc(np.load(p).copy()), tq, f"NIK-{nm}")
        r["haarpsi"] = max(x["haarpsi"] for x in fr if x["method"] == f"NIK-{nm}")
        ph.append(r)
f = f"{SW}/v2_G{G_SEL:02d}.npy"; v = np.load(f); nG = v.shape[-1]
tG = np.array([tq[g*G_SEL:(g+1)*G_SEL].mean() for g in range(nG)])
r = ph_score(sc(v.copy()), tG, f"GRASP-v2 {5*G_SEL}spf lam{LAM_SEL:g}")
r["haarpsi"] = [x for x in fr if x["method"] == "GRASP-v2" and x["spf"] == 5*G_SEL][0]["haarpsi"]
ph.append(r)
out["phantom"] = ph
print(f"PHANTOM vs truth (single setting: {5*G_SEL} spokes/frame, lam {LAM_SEL})")
print(f"{'method':22} {'frames':>7} {'HaarPSI':>8} {'aorta':>8} {'cortex':>8} {'medulla':>8} {'peak':>7} {'err%':>7}")
for r in ph:
    print(f"{r['method']:22} {r['frames']:>7} {r['haarpsi']:>8.4f} {r['aorta']:>8.4f} {r['cortex']:>8.4f} "
          f"{r['medulla']:>8.4f} {r['peak']:>7.4f} {r['peak_err_pct']:>+7.1f}")
print(f"  truth aorta peak {float(Tr[Rz['aorta']].mean(0).max()):.4f}")

# ---------------- in vivo, vs model-free nufft, CURVES ONLY ----------------
ctx = C.slice_ctx(21); rois = ctx["rois"]; bodyv = ctx["BODY"]
z = np.load(f"{B}/step2_slice21.npz")
mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = z["tmf"].astype(np.float64)
TA, NTV = 375.0, 1710
ROIS = [r for r in ("aorta", "cortex", "medulla") if rois.get(r) is not None and rois[r].sum() > 0]
def bs(c): return c - np.median(c[:8])
mfc = {r: bs(np.array([mf[..., i][rois[r]].mean() for i in range(mf.shape[-1])])) for r in ROIS}
def ft(nt): e = np.linspace(0, TA, nt+1); return 0.5*(e[:-1]+e[1:])
def refwin(nt):
    e = np.linspace(0, TA, nt+1)
    o = np.zeros((mf.shape[0], mf.shape[1], nt), np.float32)
    for g in range(nt):
        m = (tmf >= e[g]) & (tmf < e[g+1])
        if not m.any():
            m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf-0.5*(e[g]+e[g+1])))] = True
        o[:, :, g] = mf[:, :, m].mean(2)
    return o

def iv_score(v, t, lab):
    R = refwin(v.shape[-1])                      # same global LS scale both methods get
    v = v * (np.sum(v[bodyv]*R[bodyv]) / (np.sum(v[bodyv]**2) + 1e-12))
    r = {"method": lab, "frames": int(v.shape[-1])}
    for roi in ROIS:
        c = np.interp(tmf, t, np.array([v[..., i][rois[roi]].mean() for i in range(v.shape[-1])]))
        r[roi] = float(np.linalg.norm(bs(c)-mfc[roi])/np.linalg.norm(mfc[roi]))
    ca = np.interp(tmf, t, np.array([v[..., i][rois["aorta"]].mean() for i in range(v.shape[-1])]))
    r["peak_ratio"] = float(bs(ca).max()/mfc["aorta"].max()); return r
iv = []
# FAIR pair: both on the same 1368 spokes (v%10<8), NIK with heldout early stopping + LR schedule
nik = np.abs(np.load(f"{B}/results_sl21_k80/nik_slice_21.npy")).astype(np.float32)
iv.append(iv_score(nik, ft(nik.shape[-1]), "NIK k80 (1368 spokes, heldout ES)"))
gf = f"{GV}/gv2_slice21_n{NLINE_SEL:02d}_k80.npy"
if os.path.exists(gf):
    v = np.abs(np.load(gf)).astype(np.float32)
    iv.append(iv_score(v, ft(v.shape[-1]), f"GRASP-v2 NLINE {NLINE_SEL} k80 lam{LAM_SEL:g}"))
out["invivo"] = iv
print(f"\nIN VIVO vs model-free nufft, CURVES ONLY (haarpsi excluded, not informative)")
print(f"  setting carried from phantom: NLINE {NLINE_SEL} = 2.63 s/frame")
print(f"{'method':30} {'frames':>7} " + " ".join(f"{r:>9}" for r in ROIS) + f" {'pk/ref':>7}")
for r in iv:
    print(f"{r['method']:30} {r['frames']:>7} " + " ".join(f"{r[x]:>9.4f}" for x in ROIS) + f" {r['peak_ratio']:>7.4f}")
json.dump(out, open(f"{SW}/final_single_setting.json", "w"), indent=1)
print("\nFINAL_DONE")
