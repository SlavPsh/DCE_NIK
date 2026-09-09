"""phantom deliverable: grasp v2 at its BEST spokes/frame (for lam 0.25 and 0.02) vs NIK vs truth.

the knob under study is SPOKES/FRAME. lam is shown at only two fixed values, the published 0.25 and
0.02, to show whether the conclusion depends on it. best spokes/frame per lam is chosen by distance
to the ideal corner on the (haarpsi, aorta-nrmse) plane, stated so it is not a per-metric cherry pick.

out: figures/fig_v2_vs_nik.png  (images at 3 phases, contrast curves, spokes/frame sweep)
     v2_sweep/v2_vs_nik.json    (metrics table)
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, glob
import numpy as np
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import xph_pipeline as P, xph_common as X
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SW = f"{P.OUT}/v2_sweep"; A = f"{P.OUT}/arrays"; FIG = f"{P.OUT}/figures"
d = P.data(); tq = d["times"]; body = d["labels"] > 0
Rz = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq); F = len(tq)

def load_sweep():
    rows = []
    for f in sorted(glob.glob(f"{SW}/v2_G*.json")):
        r = json.load(open(f)); r.setdefault("lam_frac", 0.25)
        r["npy"] = f[:-5] + ".npy"; rows.append(r)
    return rows

def best_for(rows, lam):
    c = [r for r in rows if abs(r["lam_frac"] - lam) < 1e-9]
    if not c: return None
    return min(c, key=lambda r: (1 - r["haarpsi"])**2 + r["c_aorta"]**2)   # closest to ideal corner

def curves(v, t, roi):
    c = np.array([v[..., i][Rz[roi]].mean() for i in range(v.shape[-1])])
    return np.interp(tq, t, c)

def scale_to_truth(v):
    n = min(v.shape[-1], F)
    Tw = np.stack([Tr[:, :, i*F//v.shape[-1]] for i in range(v.shape[-1])], -1) if v.shape[-1] != F else Tr
    s = np.sum(v[body] * Tw[body]) / (np.sum(v[body]**2) + 1e-12)
    return v * s

rows = load_sweep()
nikf = {n: np.load(f"{A}/nik_fine_{n}.npy") for n in ("sub16", "free") if os.path.exists(f"{A}/nik_fine_{n}.npy")}
sel = {lam: best_for(rows, lam) for lam in (0.25, 0.02)}
print("best spokes/frame per lam (closest to ideal corner on haarpsi vs aorta-nrmse):")
for lam, r in sel.items():
    if r: print(f"  lam {lam:<5}: {r['spokes_per_frame']} spokes/frame, {r['n_frames']} frames, "
                f"haarpsi {r['haarpsi']:.4f}, aortaC {r['c_aorta']:.4f}, peak {r['aorta_pk']:.4f}")

# ---------- assemble methods ----------
M = [("truth", Tr, tq)]
for n, v in nikf.items():
    M.append((f"NIK-{n}", scale_to_truth(v.copy()), tq))
for lam, r in sel.items():
    if not r: continue
    v = np.load(r["npy"]); tG = np.array([tq[g*r["G"]:(g+1)*r["G"]].mean() for g in range(v.shape[-1])])
    M.append((f"GRASP-v2 {r['spokes_per_frame']}spf lam{lam:g}", scale_to_truth(v.copy()), tG))

TP = float(Tr[Rz["aorta"]].mean(0).max())
tab = []
for nm, v, t in M:
    row = dict(method=nm, frames=int(v.shape[-1]))
    for roi in ("aorta", "cortex", "medulla"):
        c = curves(v, t, roi); ct = Tr[Rz[roi]].mean(0)
        row[f"{roi}_nrmse"] = float(np.linalg.norm(c-ct)/(np.linalg.norm(ct)+1e-12))
    ca = curves(v, t, "aorta")
    row["aorta_peak"] = float(ca.max()); row["aorta_peak_err_pct"] = 100*(ca.max()-TP)/TP
    row["aorta_ttp"] = float(tq[np.argmax(ca)])
    tab.append(row)
print(f"\n{'method':28} {'frames':>7} {'aortaNRMSE':>11} {'cortex':>8} {'medulla':>8} {'peak':>7} {'err %':>7} {'ttp':>6}")
for r in tab:
    print(f"{r['method']:28} {r['frames']:>7} {r['aorta_nrmse']:>11.4f} {r['cortex_nrmse']:>8.4f} "
          f"{r['medulla_nrmse']:>8.4f} {r['aorta_peak']:>7.4f} {r['aorta_peak_err_pct']:>+7.1f} {r['aorta_ttp']:>6.1f}")
json.dump(dict(table=tab, selected={str(k): (v and {kk: v[kk] for kk in ('spokes_per_frame','n_frames','lam_frac','haarpsi','c_aorta','aorta_pk')}) for k, v in sel.items()}),
          open(f"{SW}/v2_vs_nik.json", "w"), indent=1)

# ---------- figure ----------
PH = [("pre ~10s", 10.0), ("peak ~27s", 27.4), ("late ~120s", 120.0)]
nm_ = len(M)
fig = plt.figure(figsize=(3.0*nm_, 10.4))
gs = fig.add_gridspec(4, nm_, height_ratios=[1, 1, 1, 1.35], hspace=0.16, wspace=0.04)
vmax = float(np.percentile(Tr[body], 99.5))
for pi, (plab, pt) in enumerate(PH):
    for mi, (nm, v, t) in enumerate(M):
        ax = fig.add_subplot(gs[pi, mi]); i = int(np.argmin(np.abs(t - pt)))
        ax.imshow(v[:, :, i], cmap="gray", vmin=0, vmax=vmax); ax.axis("off")
        if pi == 0: ax.set_title(nm, fontsize=8.5)
        if mi == 0: ax.text(-0.08, 0.5, plab, transform=ax.transAxes, rotation=90, va="center", fontsize=9)
COL = {"truth": "k"}
for mi, (nm, v, t) in enumerate(M):
    COL.setdefault(nm, ["k", "#7c3aed", "#0369a1", "#c0392b", "#e67e22"][mi % 5])
for ri, roi in enumerate(("aorta", "cortex", "medulla")):
    ax = fig.add_subplot(gs[3, ri*nm_//3:(ri+1)*nm_//3] if nm_ >= 3 else gs[3, :])
    for nm, v, t in M:
        c = curves(v, t, roi)
        ax.plot(tq, c, color=COL[nm], lw=2.2 if nm == "truth" else 1.5,
                ls="-" if nm == "truth" else "--", label=nm)
    ax.set_title(f"{roi} enhancement", fontsize=9); ax.set_xlabel("time (s)", fontsize=8)
    ax.grid(alpha=.3); ax.tick_params(labelsize=7)
    if ri == 0: ax.legend(fontsize=6.2, loc="upper right")
fig.suptitle("phantom (xcat no-motion) vs ground truth. grasp v2 single setting = 25 spf lam 0.25 (all decisions); 40 spf lam 0.02 = lam contrast only.\n"
             "all methods on the same 5 of 7 angles/frame (train_ang, 71% of acquired), globally scaled to truth",
             fontsize=10)
fig.savefig(f"{FIG}/fig_v2_vs_nik.png", dpi=125, bbox_inches="tight")
print(f"\nSAVED {FIG}/fig_v2_vs_nik.png")
print("V2_VS_NIK_DONE")
