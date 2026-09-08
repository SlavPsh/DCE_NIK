"""GRASP-Pro K-sweep on the XCAT phantom (z15): trace the spatial<->temporal Pareto frontier of the
temporal-PCA subspace rank K. Reconstruct at each K (library NLCG + NUFFT-SENSE true-coil operator),
compute the SAME spatial (SSIM/PSNR/HaarPSI/NRMSE) and temporal (aorta/cortex/medulla curve-NRMSE,
aorta first-pass peak/FWHM) metrics as xph_aggregate, overlay the NIK variants + CS-file, and plot the
frontier. Answers: does NIK sit outside GRASP's reachable frontier (NIK dominates) or inside it."""
import warnings; warnings.filterwarnings("ignore")
import os, glob, json, csv, re, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_grasp_nufft as GN, xph_pipeline as P, xph_common as X

KS_LIST = [3, 5, 8, 12, 16, 24, 32]
OUT = X.OUT; FIG = f"{OUT}/figures"; A = f"{OUT}/arrays"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
rv = float(Tr[body].max() - Tr[body].min()); _p99 = np.percentile(Tr[body], 99) + 1e-9
_ys, _xs = np.where(body); _y0, _y1, _x0, _x1 = _ys.min(), _ys.max()+1, _xs.min(), _xs.max()+1

# ---- metrics identical to xph_aggregate ----
from scipy.ndimage import uniform_filter
import torch as _t, piq
def mnr(a, b): return float(np.sqrt(np.mean((a[body]-b[body])**2))/(rv+1e-12))
def ssim(a, b, win=7):
    C1 = (0.01*rv)**2; C2 = (0.03*rv)**2; ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    va = uniform_filter(a*a, win)-ma**2; vb = uniform_filter(b*b, win)-mb**2; vab = uniform_filter(a*b, win)-ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2)))[body].mean())
def psnr(a, b): return float(20*np.log10(rv/(np.sqrt(np.mean((a[body]-b[body])**2))+1e-12)))
def haar(a, b):
    xa = _t.tensor(np.clip(a[_y0:_y1, _x0:_x1]/_p99, 0, 1), dtype=_t.float32)[None, None]
    xb = _t.tensor(np.clip(b[_y0:_y1, _x0:_x1]/_p99, 0, 1), dtype=_t.float32)[None, None]
    try: return float(piq.haarpsi(xa, xb, data_range=1.0).item())
    except Exception: return float("nan")
def per_frame(rec, fn): return np.array([fn(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])
_pre = tq < 18
def _bsub(c): return c - np.median(c[_pre])
truec = {nm: _bsub(np.median(Tr[R[nm]], 0)) for nm in ["aorta", "cortex", "medulla"]}
def curve_nrmse(rec, nm): return float(np.linalg.norm(_bsub(np.median(rec[R[nm]], 0))-truec[nm])/np.linalg.norm(truec[nm]))
def fp_metrics(c):
    b = c[:20].mean(); n = c - b; pk = n.max(); ttp = tq[n.argmax()]; idx = np.where((n >= pk/2) & (tq < ttp+50))[0]
    return float(ttp), float(tq[idx[-1]]-tq[idx[0]]) if idx.size > 1 else 0.0, float(c.max())
def all_metrics(rec):
    m = dict(SSIM=float(per_frame(rec, ssim).mean()), PSNR=float(np.nanmean(per_frame(rec, psnr))),
             HaarPSI=float(np.nanmean(per_frame(rec, haar))), NRMSE=float(per_frame(rec, mnr).mean()))
    for nm in ["aorta", "cortex", "medulla"]: m[f"curve_{nm}"] = curve_nrmse(rec, nm)
    fp = fp_metrics(_bsub(np.median(rec[R["aorta"]], 0))); m["aorta_ttp"], m["aorta_fwhm"], m["aorta_peak"] = fp
    return m

def orient(v, nm): return {"id": v, "rot180": v[::-1, ::-1], "fliplr": v[:, ::-1], "flipud": v[::-1], "T": np.transpose(v, (1, 0, 2)), "rot90": np.rot90(v), "rot270": np.rot90(v, 3)}[nm]
def cc(a, b): a = a[body].ravel()-a[body].mean(); b = b[body].ravel()-b[body].mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
def orient_scale(rec0, tm):
    best = max(["id", "rot180", "fliplr", "flipud", "T", "rot90", "rot270"], key=lambda nm: cc(orient(rec0, nm).mean(2), tm))
    rec = orient(rec0, best); s = np.sum(rec[body]*Tr[body])/(np.sum(rec[body]**2)+1e-12)
    return rec*s, best, float(s)
tm = Tr.mean(2)

# ---- GRASP K-sweep ----
rows = []
for K in KS_LIST:
    dyn, Phi = GN.reconstruct(k=K); rec, best, s = orient_scale(np.abs(dyn), tm)
    np.savez(f"{A}/grasp_ksweep_K{K}.npz", rec=rec.astype(np.float32), K=K, orient=best, scale=s)
    m = all_metrics(rec); m.update(method=f"GRASP-K{K}", family="GRASP", K=K)
    rows.append(m); print(f"[K={K:2d}] orient={best} SSIM {m['SSIM']:.3f} PSNR {m['PSNR']:.1f} Haar {m['HaarPSI']:.3f} | aortaCurve {m['curve_aorta']:.3f} peak {m['aorta_peak']:.3f} fwhm {m['aorta_fwhm']:.1f}", flush=True)

# ---- NIK variants + CS-file overlay (loaded + scaled exactly like xph_aggregate) ----
sel = json.load(open(f"{A}/stageA_selection.json")); W = sel["selected_width"]
KSIG = json.load(open(f"{A}/stageB_selection.json"))["selected_ks"] if os.path.exists(f"{A}/stageB_selection.json") else 2.5
NIK = {}
for f in sorted(glob.glob(f"{A}/nik_eval_w{W}_ks{KSIG:g}_s*.npz")): NIK.setdefault("NIK-F0", []).append(np.load(f)["rec_best"])
for f in sorted(glob.glob(f"{A}/img_eval_sub*_w{W}_s*.npz")):
    Rk = re.search(r"sub(\d+)_w", f).group(1); NIK.setdefault(f"NIK-sub{Rk}", []).append(np.load(f)["rec_best"])
for f in sorted(glob.glob(f"{A}/img_eval_free_w{W}_s*.npz")): NIK.setdefault("NIK-free", []).append(np.load(f)["rec_best"])
def repr_recon(recs):  # median-NRMSE seed, then LS-scale
    sc = [r*float((r[body]*Tr[body]).sum()/((r[body]**2).sum()+1e-9)) for r in recs]
    nr = [per_frame(r, mnr).mean() for r in sc]; return sc[int(np.argsort(nr)[len(nr)//2])]
for nm, recs in NIK.items():
    m = all_metrics(repr_recon(recs)); m.update(method=nm, family="NIK", K=np.nan); rows.append(m)
    print(f"[{nm}] SSIM {m['SSIM']:.3f} Haar {m['HaarPSI']:.3f} | aortaCurve {m['curve_aorta']:.3f} peak {m['aorta_peak']:.3f}", flush=True)
try:
    import h5py
    rf = h5py.File(X.SIM, "r"); rc = np.abs(np.array(rf["results"]["images"]["Recon"]["img"])[:, P.ZI]).astype(np.float32)
    csf = np.stack([X._embed(rc[i], rc.shape[1]) for i in range(rc.shape[0])], -1)
    if cc(csf.mean(2), tm) < cc(csf[::-1, ::-1].mean(2), tm): csf = csf[::-1, ::-1]
    if csf.shape[2] == len(tq):
        csf = csf*float((csf[body]*Tr[body]).sum()/((csf[body]**2).sum()+1e-9))
        m = all_metrics(csf); m.update(method="CS-file", family="CS", K=np.nan); rows.append(m)
        print(f"[CS-file] SSIM {m['SSIM']:.3f} Haar {m['HaarPSI']:.3f} | aortaCurve {m['curve_aorta']:.3f}", flush=True)
except Exception as e: print("CS-file skipped:", str(e)[:100])

# ---- CSV ----
keys = ["method", "family", "K", "SSIM", "PSNR", "HaarPSI", "NRMSE", "curve_aorta", "curve_cortex", "curve_medulla", "aorta_peak", "aorta_fwhm", "aorta_ttp"]
with open(f"{OUT}/grasp_ksweep_pareto.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(keys)
    for r in rows: w.writerow([r.get(k, "") for k in keys])

# ---- Pareto figure: spatial (y) vs temporal (x); GRASP frontier + NIK/CS points ----
G = [r for r in rows if r["family"] == "GRASP"]; G = sorted(G, key=lambda r: r["K"])
others = [r for r in rows if r["family"] != "GRASP"]
cols = {"NIK": "C3", "CS": "C1"}
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
panels = [("SSIM", "curve_aorta", "aorta curve-nrmse (lower better)", "ssim (higher better)"),
          ("HaarPSI", "curve_aorta", "aorta curve-nrmse (lower better)", "haarpsi (higher better)"),
          ("SSIM", "aorta_peak", "aorta first-pass peak (truth 0.77)", "ssim (higher better)")]
for a, (yk, xk, xl, yl) in zip(ax, panels):
    gx = [r[xk] for r in G]; gy = [r[yk] for r in G]
    a.plot(gx, gy, "-o", color="C0", label="GRASP K-sweep", zorder=3)
    for r in G: a.annotate(f"K{r['K']}", (r[xk], r[yk]), fontsize=7, color="C0", xytext=(3, 3), textcoords="offset points")
    for r in others:
        a.scatter(r[xk], r[yk], color=cols.get(r["family"], "C4"), zorder=4)
        a.annotate(r["method"], (r[xk], r[yk]), fontsize=7, xytext=(3, -8), textcoords="offset points")
    if xk == "aorta_peak": a.axvline(0.772, color="k", ls=":", lw=1, alpha=0.6)
    a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.2)
ax[0].legend(fontsize=8, loc="lower left")
fig.suptitle("grasp-pro spatial<->temporal frontier (pca rank K) vs nik. z15 phantom, vs truth")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_ksweep_pareto.png", dpi=130); plt.close(fig)
print("SAVED", f"{FIG}/fig_ksweep_pareto.png", "+ grasp_ksweep_pareto.csv"); print("KSWEEP_DONE")
