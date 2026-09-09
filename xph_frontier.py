"""spatial-vs-temporal frontier: classic GRASP v2 (swept over spokes/frame) vs NIK.

the point of the sweep is to give grasp v2 its BEST operating point rather than the arbitrary
5 spokes/frame it was pinned at. both methods are scored on IDENTICAL rulers at every G:
  spatial  = haarpsi vs truth averaged over the same window the frame integrates
  temporal = aorta curve nrmse on the fine truth grid
NIK is binning-free, so its "frame" for a window is the time-average of its continuous render over
that window, the direct analogue of what grasp's data integrates.

dominance is reported criterion-free: a method dominates if it is better on BOTH axes at the
comparison point. that avoids inventing a spatial-vs-temporal weighting.
out: v2_sweep/frontier.json + figures/fig_v2_frontier.png
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, glob
import numpy as np, torch
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import xph_pipeline as P, xph_common as X
from masked_metrics import haarpsi_masked, ssim_masked
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

A = f"{P.OUT}/arrays"; SW = f"{P.OUT}/v2_sweep"; FIG = f"{P.OUT}/figures"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = P.data(); tq = d["times"]; body = d["labels"] > 0
Rz = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq); F = len(tq)
mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev)

def window_avg(v, G):
    nG = v.shape[-1] // G
    return np.stack([v[:, :, g*G:(g+1)*G].mean(2) for g in range(nG)], -1)

def spatial(rec, TrW):
    nG = TrW.shape[-1]
    s = np.sum(rec[body]*TrW[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s
    hs, ss = [], []
    for t in range(0, nG, max(1, nG//40)):
        vmax = float(np.percentile(TrW[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        rt = torch.from_numpy(np.clip(TrW[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu()))
        ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))
    rv = float(TrW[body].max()-TrW[body].min())
    nr = float(np.mean([np.sqrt(np.mean((rec[:, :, t][body]-TrW[:, :, t][body])**2))/rv for t in range(nG)]))
    return float(np.mean(hs)), float(np.mean(ss)), nr, rec

def temporal(rec, G):
    """aorta curve nrmse on the FINE grid (coarse curves interpolated up -> binning is charged here)."""
    nG = rec.shape[-1]
    tG = np.array([tq[g*G:(g+1)*G].mean() for g in range(nG)])
    out = {}
    for nm in ("aorta", "cortex", "medulla"):
        cg = rec[Rz[nm]].mean(0); ct = Tr[Rz[nm]].mean(0)
        ci = np.interp(tq, tG, cg)
        out[f"c_{nm}"] = float(np.linalg.norm(ci-ct)/(np.linalg.norm(ct)+1e-12))
    ca = np.interp(tq, tG, rec[Rz["aorta"]].mean(0))
    out["aorta_pk"] = float(ca.max()); out["aorta_ttp"] = float(tq[np.argmax(ca)])
    return out

rows = []
niks = {n: np.load(f"{A}/nik_fine_{n}.npy") for n in ("sub16", "free") if os.path.exists(f"{A}/nik_fine_{n}.npy")}
for f in sorted(glob.glob(f"{SW}/v2_G*.npy")):
    G = int(os.path.basename(f)[4:6])
    rec = np.load(f); TrW = window_avg(Tr, G)
    h, s_, nr, recs = spatial(rec, TrW); t = temporal(recs, G)
    rows.append(dict(method="GRASP-v2", G=G, spf=5*G, frames=rec.shape[-1],
                     dt=float(np.diff(tq).mean()*G), haarpsi=h, ssim=s_, nrmse=nr, **t))
    for n, v in niks.items():
        vg = window_avg(v, G)
        h2, s2, nr2, recs2 = spatial(vg, TrW); t2 = temporal(recs2, G)
        rows.append(dict(method=f"NIK-{n}", G=G, spf=5*G, frames=vg.shape[-1],
                         dt=float(np.diff(tq).mean()*G), haarpsi=h2, ssim=s2, nrmse=nr2, **t2))

json.dump(rows, open(f"{SW}/frontier.json", "w"), indent=1)
print(f"{'method':12} {'spf':>4} {'frames':>7} {'dt(s)':>6} {'HaarPSI':>8} {'SSIM':>7} {'aortaC':>7} {'aortaPk':>8}")
for r in sorted(rows, key=lambda z: (z["method"], z["G"])):
    print(f"{r['method']:12} {r['spf']:>4} {r['frames']:>7} {r['dt']:>6.2f} {r['haarpsi']:>8.4f} {r['ssim']:>7.4f} {r['c_aorta']:>7.4f} {r['aorta_pk']:>8.4f}")

# ---- grasp's own best operating point + dominance ----
g = [r for r in rows if r["method"] == "GRASP-v2"]
print(f"\ntruth aorta peak {Tr[Rz['aorta']].mean(0).max():.4f}")
best_sp = max(g, key=lambda r: r["haarpsi"]); best_tp = min(g, key=lambda r: r["c_aorta"])
print(f"grasp v2 best SPATIAL : spf={best_sp['spf']} haarpsi={best_sp['haarpsi']:.4f} aortaC={best_sp['c_aorta']:.4f}")
print(f"grasp v2 best TEMPORAL: spf={best_tp['spf']} haarpsi={best_tp['haarpsi']:.4f} aortaC={best_tp['c_aorta']:.4f}")
# clinically reasonable renal DCE temporal resolution is ~2-5 s/frame. declared BEFORE looking at
# nik, and it is also where grasp v2's own combined optimum sits, so it is not a nik-flattering band.
BAND = (2.0, 5.0)
print(f"\nreasonable-recon band: {BAND[0]}-{BAND[1]} s/frame")
for n in niks:
    nk = {r["G"]: r for r in rows if r["method"] == f"NIK-{n}"}
    print(f"\n  NIK-{n} vs grasp v2, matched spokes/frame:")
    print(f"    {'spf':>4} {'dt':>5} | {'haarpsi g':>9} {'haarpsi n':>9} {'d':>7} | {'aortaC g':>8} {'aortaC n':>8} {'d':>7}  {'in band':>8}")
    for r in sorted(g, key=lambda z: z["G"]):
        q = nk.get(r["G"])
        if q is None: continue
        inb = "yes" if BAND[0] <= r["dt"] <= BAND[1] else ""
        print(f"    {r['spf']:>4} {r['dt']:>5.2f} | {r['haarpsi']:>9.4f} {q['haarpsi']:>9.4f} {q['haarpsi']-r['haarpsi']:>+7.4f}"
              f" | {r['c_aorta']:>8.4f} {q['c_aorta']:>8.4f} {q['c_aorta']-r['c_aorta']:>+7.4f}  {inb:>8}")
    inband = [r for r in g if BAND[0] <= r["dt"] <= BAND[1] and r["G"] in nk]
    wins_both = [r for r in inband if nk[r["G"]]["haarpsi"] > r["haarpsi"] and nk[r["G"]]["c_aorta"] < r["c_aorta"]]
    wins_sp   = [r for r in inband if nk[r["G"]]["haarpsi"] > r["haarpsi"]]
    wins_tp   = [r for r in inband if nk[r["G"]]["c_aorta"] < r["c_aorta"]]
    print(f"    -> in band: nik better spatially at {len(wins_sp)}/{len(inband)}, "
          f"temporally at {len(wins_tp)}/{len(inband)}, both at {len(wins_both)}/{len(inband)}")
    # headline: nik unbinned (no tradeoff to make) vs grasp at ITS OWN best combined point
    if 1 in nk:
        gb = min(g, key=lambda r: (1 - r["haarpsi"]) ** 2 + r["c_aorta"] ** 2)   # closest to ideal corner
        print(f"    -> grasp v2 best combined = {gb['spf']} spf ({gb['dt']:.1f}s): haarpsi {gb['haarpsi']:.4f} aortaC {gb['c_aorta']:.4f}")
        print(f"       nik-{n} unbinned (0.52s)        : haarpsi {nk[1]['haarpsi']:.4f} aortaC {nk[1]['c_aorta']:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
LEG = {"GRASP-v2": "GRASP-v2 classic 2014: MCNUFFT + temporal TV, no subspace, lam=0.25max|x0|",
       "NIK-sub16": "NIK-sub16: wire_ff_subspace rank 16, 2-seed cplx avg",
       "NIK-free":  "NIK-free: wire_ff full rank, 3-seed cplx avg"}
for m, c in (("GRASP-v2", "#c0392b"), ("NIK-sub16", "#7c3aed"), ("NIK-free", "#0369a1")):
    r = sorted([x for x in rows if x["method"] == m], key=lambda z: z["G"])
    if not r: continue
    ax.plot([x["c_aorta"] for x in r], [x["haarpsi"] for x in r], "o-", color=c, label=LEG[m], lw=2, ms=6)
    for x in r: ax.annotate(f"{x['spf']}", (x["c_aorta"], x["haarpsi"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("aorta curve nrmse, fine grid (temporal, lower better)")
ax.set_ylabel("haarpsi vs window-averaged truth (spatial, higher better)")
ax.set_title("spatial-temporal frontier, phantom (xcat no-motion, vs truth)\nALL methods on the SAME 5 of 7 angles/frame (TRAIN_ANG, 71% of acquired)\npoint labels = spokes/frame after grouping; up and left is better", fontsize=9)
ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower left"); fig.tight_layout()
fig.savefig(f"{FIG}/fig_v2_frontier.png", dpi=130); print(f"\nSAVED {FIG}/fig_v2_frontier.png")
print("FRONTIER_DONE")
