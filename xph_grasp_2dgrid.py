"""GRASP-Pro 2D (K x spokes-per-frame) grid on the XCAT phantom (z15). Second Pareto axis: hold the
training-spoke pool fixed (5 angles x 344 frames = 1720 spokes, acquisition order), regroup into frames
of SPF spokes each (nframes = 1720//SPF), rebuild the temporal-PCA navigator + operator on that grid,
reconstruct, then interpolate the dynamic back onto the 344-frame truth grid so the SAME metrics as the
K-sweep apply. Traces the full reachable GRASP frontier; overlaid with the K-sweep line + NIK points."""
import warnings; warnings.filterwarnings("ignore")
import os, glob, json, csv, re, copy, numpy as np, cupy as cp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import xph_grasp_nufft as GN, xph_pipeline as P, xph_common as X
import precompute_ref as pr

SPF_LIST = [2, 5, 10, 20]; K_LIST = [5, 12, 24]
OUT = X.OUT; FIG = f"{OUT}/figures"; A = f"{OUT}/arrays"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
rv = float(Tr[body].max() - Tr[body].min()); _p99 = np.percentile(Tr[body], 99) + 1e-9
_ys, _xs = np.where(body); _y0, _y1, _x0, _x1 = _ys.min(), _ys.max()+1, _xs.min(), _xs.max()+1
tm = Tr.mean(2)

# ---- metrics (identical to xph_aggregate / xph_grasp_ksweep) ----
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

# ---- rebin the training-spoke stream to SPF spokes/frame ----
kx = d["kx"]; ky = d["ky"]; kdata = d["kdata"]; b1 = d["b1"]; C, F, nang, RO = kdata.shape; tr = np.array(P.TRAIN_ANG)
dt = np.gradient(tq)                                                   # per-frame spacing
sx = []; sy = []; sk = []; st = []                                     # per training-spoke: traj + data + time (acq order)
for fr in range(F):
    for a in tr:
        sx.append(kx[fr, a]); sy.append(ky[fr, a]); sk.append(kdata[:, fr, a, :])
        st.append(tq[fr] + (a - (nang-1)/2.0)/nang * dt[fr])
sx = np.array(sx); sy = np.array(sy); sk = np.stack(sk, 1); st = np.array(st)   # sk [C, Nsp, RO]
Nsp = sx.shape[0]; c0 = RO//2
def groups_for(spf): return [np.arange(i, min(i+spf, Nsp)) for i in range(0, Nsp, spf)]
def build_phi_flat(groups, K):
    feats = np.stack([np.abs(sk[:, g, c0-2:c0+3]).mean(1).ravel() for g in groups], 1)   # [C*5, nframes]
    w, PC = np.linalg.eigh(np.cov(feats, rowvar=False)); return PC[:, np.argsort(-w)][:, :K].astype(np.complex64)

def reconstruct_spf(spf, K):
    groups = groups_for(spf); nt = len(groups); ftime = np.array([st[g].mean() for g in groups])
    trajs = [(-sx[g], -sy[g]) for g in groups]
    dcf = [np.maximum(np.abs(sx[g] + 1j*sy[g]), 1e-3) for g in groups]
    Phi = build_phi_flat(groups, K); PCA = pr.TempPCASub(Phi)
    E = GN.Emat_NUFFT(trajs, dcf, b1, Phi, RO)
    raw = np.stack([sk[:, g, :].reshape(C, -1) for g in groups], 1).astype(np.complex128)
    y = E.apply_dcf(raw); recon = E.H @ y
    for _ in range(GN.NOUTER):
        recon = pr.cs_l1_nlcg_sptv(recon, dict(E=E, y=y, PCA=PCA, TV1=pr.TV_Temp(), TV2=pr.FD1OP(),
            TVWeight1=np.abs(recon).max()*pr.Weight1, TVWeight2=np.abs(recon).max()*pr.Weight2, nite=GN.NITE))
    dyn = np.abs(np.asarray(PCA.H @ recon))                            # [RO,RO,nt]
    return dyn, ftime, nt

def to_tq(dyn, ftime):                                                 # resample native-frame dynamic onto tq
    f = interp1d(ftime, dyn, axis=2, kind="linear", bounds_error=False, fill_value=(dyn[:, :, 0], dyn[:, :, -1]))
    return f(tq).astype(np.float32)

# ---- run grid ----
rows = []
for spf in SPF_LIST:
    for K in K_LIST:
        dyn, ftime, nt = reconstruct_spf(spf, K); rec0 = to_tq(dyn, ftime)
        best = max(["id", "rot180", "fliplr", "flipud", "T", "rot90", "rot270"], key=lambda nm: cc(orient(rec0, nm).mean(2), tm))
        rec = orient(rec0, best); rec = rec*float((rec[body]*Tr[body]).sum()/((rec[body]**2).sum()+1e-12))
        m = all_metrics(rec); m.update(method=f"K{K}/spf{spf}", family="GRASP-2d", K=K, spf=spf, nframes=nt)
        rows.append(m); print(f"[spf={spf:2d} K={K:2d} nf={nt:4d}] SSIM {m['SSIM']:.3f} Haar {m['HaarPSI']:.3f} | aortaCurve {m['curve_aorta']:.3f} peak {m['aorta_peak']:.3f} fwhm {m['aorta_fwhm']:.1f}", flush=True)
        del dyn, rec0, rec; cp.get_default_memory_pool().free_all_blocks()

# ---- pull K-sweep + NIK/CS points from the K-sweep CSV for overlay ----
extra = []
kcsv = f"{OUT}/grasp_ksweep_pareto.csv"
if os.path.exists(kcsv):
    import csv as _c
    for r in _c.DictReader(open(kcsv)):
        if r["family"] in ("NIK", "CS") or r["family"] == "GRASP":
            extra.append({k: (r[k] if k in ("method", "family") else (float(r[k]) if r[k] not in ("", "nan") else np.nan)) for k in r})

keys = ["method", "family", "K", "spf", "nframes", "SSIM", "PSNR", "HaarPSI", "NRMSE", "curve_aorta", "curve_cortex", "curve_medulla", "aorta_peak", "aorta_fwhm", "aorta_ttp"]
with open(f"{OUT}/grasp_2dgrid_pareto.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(keys)
    for r in rows: w.writerow([r.get(k, "") for k in keys])

# ---- figure: full GRASP frontier (2d grid + K-sweep) vs NIK/CS ----
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
mk = {2: "o", 5: "s", 10: "^", 20: "D"}
for (yk, xk, xl, yl) in [("SSIM", "curve_aorta", "aorta curve-nrmse (lower better)", "ssim (higher better)"),
                          ("HaarPSI", "aorta_peak", "aorta first-pass peak (truth 0.77)", "haarpsi (higher better)")]:
    a = ax[0] if xk == "curve_aorta" else ax[1]
    for K in K_LIST:
        pts = sorted([r for r in rows if r["K"] == K], key=lambda r: r["spf"])
        a.plot([r[xk] for r in pts], [r[yk] for r in pts], "-", color="C0", alpha=0.4, zorder=2)
    for r in rows:
        a.scatter(r[xk], r[yk], marker=mk[r["spf"]], color="C0", s=45, zorder=3)
        a.annotate(f"K{r['K']}s{r['spf']}", (r[xk], r[yk]), fontsize=6, color="C0", xytext=(2, 2), textcoords="offset points")
    for r in extra:
        col = {"NIK": "C3", "CS": "C1", "GRASP": "C2"}[r["family"]]
        a.scatter(r[xk], r[yk], color=col, marker="*" if r["family"] == "NIK" else "P", s=90, zorder=4)
        a.annotate(r["method"], (r[xk], r[yk]), fontsize=6, xytext=(2, -9), textcoords="offset points")
    if xk == "aorta_peak": a.axvline(0.772, color="k", ls=":", lw=1, alpha=0.6)
    a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.2)
ax[0].set_title("blue=grasp 2d grid (o spf2, s spf5, ^ spf10, D spf20), green=grasp K-sweep line at spf5")
fig.suptitle("full grasp-pro frontier (K x spokes/frame) vs nik. z15 phantom, vs truth")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_2dgrid_pareto.png", dpi=130); plt.close(fig)
print("SAVED", f"{FIG}/fig_2dgrid_pareto.png", "+ grasp_2dgrid_pareto.csv"); print("GRID2D_DONE")
