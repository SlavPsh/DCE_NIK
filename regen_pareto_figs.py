"""Fix the k-sweep + 2d-grid Pareto plots: their NIK/CS overlay points were computed PRE the 1px render
fix (SSIM ~0.75 and stale temporal). The GRASP K-sweep line/grid is unaffected (GRASP was never shifted)
and is reused from the saved CSVs/recons. Recompute NIK/CS from the CORRECTED recons, rewrite the NIK/CS
rows in grasp_ksweep_pareto.csv, and redraw fig_ksweep_pareto + fig_2dgrid_pareto."""
import warnings; warnings.filterwarnings("ignore")
import glob, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
import torch as _t, piq
import xph_pipeline as P, xph_common as X
OUT = X.OUT; FIG = f"{OUT}/figures"; A = f"{OUT}/arrays"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
rv = float(Tr[body].max()-Tr[body].min()); _p99 = np.percentile(Tr[body], 99)+1e-9
_ys, _xs = np.where(body); _y0, _y1, _x0, _x1 = _ys.min(), _ys.max()+1, _xs.min(), _xs.max()+1
_pre = tq < 18
def bsub(c): return c - np.median(c[_pre])
def ls(v): return v * float((v[body]*Tr[body]).sum()/((v[body]**2).sum()+1e-12))
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
def pf(rec, fn): return np.array([fn(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])
truec = {nm: bsub(np.median(Tr[R[nm]], 0)) for nm in ["aorta", "cortex", "medulla"]}
def cnr(rec, nm): return float(np.linalg.norm(bsub(np.median(rec[R[nm]], 0))-truec[nm])/np.linalg.norm(truec[nm]))
def fp(rec):
    c = bsub(np.median(rec[R["aorta"]], 0)); b = c[:20].mean(); n = c-b; pk = n.max(); ttp = tq[n.argmax()]
    idx = np.where((n >= pk/2) & (tq < ttp+50))[0]; return float(c.max()), (float(tq[idx[-1]]-tq[idx[0]]) if idx.size > 1 else 0.0), float(ttp)
def metrics_of(rec):
    m = dict(SSIM=float(pf(rec, ssim).mean()), PSNR=float(np.nanmean(pf(rec, psnr))), HaarPSI=float(np.nanmean(pf(rec, haar))), NRMSE=float(pf(rec, mnr).mean()))
    for nm in ["aorta", "cortex", "medulla"]: m[f"curve_{nm}"] = cnr(rec, nm)
    m["aorta_peak"], m["aorta_fwhm"], m["aorta_ttp"] = fp(rec); return m
def repr_recon(pat):
    fs = sorted(glob.glob(f"{A}/{pat}.npz")); recs = [ls(np.abs(np.load(f)["rec_best"]).astype(np.float64)) for f in fs]
    return recs[int(np.argsort([pf(r, mnr).mean() for r in recs])[len(recs)//2])]

# --- corrected NIK/CS points ---
NIKPATS = {"NIK-F0": "nik_eval_w768_ks2.5_s*", "NIK-sub5": "img_eval_sub5_w768_s*", "NIK-sub12": "img_eval_sub12_w768_s*",
           "NIK-sub16": "img_eval_sub16_w768_s*", "NIK-free": "img_eval_free_w768_s*"}
rows_new = {}
for nm, pat in NIKPATS.items(): rows_new[nm] = dict(metrics_of(repr_recon(pat)), method=nm, family="NIK", K="nan")
try:
    import h5py; rf = h5py.File(X.SIM, "r"); rc = np.abs(np.array(rf["results"]["images"]["Recon"]["img"])[:, P.ZI]).astype(np.float32)
    csf = np.stack([X._embed(rc[i], rc.shape[1]) for i in range(rc.shape[0])], -1)
    def cc(a, b): a = a[body].ravel()-a[body].mean(); b = b[body].ravel()-b[body].mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    if cc(csf.mean(2), Tr.mean(2)) < cc(csf[::-1, ::-1].mean(2), Tr.mean(2)): csf = csf[::-1, ::-1]
    rows_new["CS-file"] = dict(metrics_of(ls(csf.astype(np.float64))), method="CS-file", family="CS", K="nan")
except Exception as e: print("CS-file skipped:", str(e)[:80])

# --- rewrite grasp_ksweep_pareto.csv: keep GRASP rows, replace NIK/CS ---
keys = ["method", "family", "K", "SSIM", "PSNR", "HaarPSI", "NRMSE", "curve_aorta", "curve_cortex", "curve_medulla", "aorta_peak", "aorta_fwhm", "aorta_ttp"]
old = list(csv.DictReader(open(f"{OUT}/grasp_ksweep_pareto.csv")))
grasp_rows = [r for r in old if r["family"] == "GRASP"]
with open(f"{OUT}/grasp_ksweep_pareto.csv", "w", newline="") as fp2:
    w = csv.writer(fp2); w.writerow(keys)
    for r in grasp_rows: w.writerow([r.get(k, "") for k in keys])
    for nm, r in rows_new.items(): w.writerow([r.get(k, "") for k in keys])
print("updated grasp_ksweep_pareto.csv NIK/CS rows:", {nm: round(rows_new[nm]["SSIM"], 3) for nm in rows_new})

# --- redraw fig_ksweep_pareto (3 panels) ---
def num(r, k):
    try: return float(r[k])
    except Exception: return np.nan
rows = list(csv.DictReader(open(f"{OUT}/grasp_ksweep_pareto.csv")))
G = sorted([r for r in rows if r["family"] == "GRASP"], key=lambda r: num(r, "K")); others = [r for r in rows if r["family"] != "GRASP"]
cols = {"NIK": "C3", "CS": "C1"}
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
panels = [("SSIM", "curve_aorta", "aorta curve-nrmse (lower better)", "ssim (higher better)"),
          ("HaarPSI", "curve_aorta", "aorta curve-nrmse (lower better)", "haarpsi (higher better)"),
          ("SSIM", "aorta_peak", "aorta first-pass peak (truth 0.77)", "ssim (higher better)")]
for a, (yk, xk, xl, yl) in zip(ax, panels):
    a.plot([num(r, xk) for r in G], [num(r, yk) for r in G], "-o", color="C0", label="GRASP K-sweep", zorder=3)
    for r in G: a.annotate(f"K{int(num(r,'K'))}", (num(r, xk), num(r, yk)), fontsize=7, color="C0", xytext=(3, 3), textcoords="offset points")
    for r in others:
        a.scatter(num(r, xk), num(r, yk), color=cols.get(r["family"], "C4"), zorder=4)
        a.annotate(r["method"], (num(r, xk), num(r, yk)), fontsize=7, xytext=(3, -8), textcoords="offset points")
    if xk == "aorta_peak": a.axvline(0.772, color="k", ls=":", lw=1, alpha=0.6)
    a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.2)
ax[0].legend(fontsize=8, loc="lower left")
fig.suptitle("grasp-pro spatial<->temporal frontier (pca rank K) vs nik (POST 1px-fix). z15 phantom, vs truth")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_ksweep_pareto.png", dpi=130); plt.close(fig)

# --- redraw fig_2dgrid_pareto (2 panels): GRASP-2d grid + K-sweep line + corrected NIK/CS ---
g2 = list(csv.DictReader(open(f"{OUT}/grasp_2dgrid_pareto.csv"))); mk = {2: "o", 5: "s", 10: "^", 20: "D"}
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for (yk, xk, xl, yl), a in zip([("SSIM", "curve_aorta", "aorta curve-nrmse (lower better)", "ssim (higher better)"),
                                 ("HaarPSI", "aorta_peak", "aorta first-pass peak (truth 0.77)", "haarpsi (higher better)")], ax):
    for K in sorted(set(int(num(r, "K")) for r in g2)):
        pts = sorted([r for r in g2 if int(num(r, "K")) == K], key=lambda r: num(r, "spf"))
        a.plot([num(r, xk) for r in pts], [num(r, yk) for r in pts], "-", color="C0", alpha=0.4, zorder=2)
    for r in g2:
        a.scatter(num(r, xk), num(r, yk), marker=mk.get(int(num(r, "spf")), "x"), color="C0", s=45, zorder=3)
    for r in G: a.scatter(num(r, xk), num(r, yk), color="C2", marker="P", s=70, zorder=4)          # GRASP K-sweep (spf5) line
    for r in others:
        col = {"NIK": "C3", "CS": "C1"}[r["family"]]
        a.scatter(num(r, xk), num(r, yk), color=col, marker="*" if r["family"] == "NIK" else "P", s=110, zorder=5)
        a.annotate(r["method"], (num(r, xk), num(r, yk)), fontsize=7, xytext=(2, -9), textcoords="offset points")
    if xk == "aorta_peak": a.axvline(0.772, color="k", ls=":", lw=1, alpha=0.6)
    a.set_xlabel(xl); a.set_ylabel(yl); a.grid(alpha=0.2)
ax[0].set_title("blue=grasp 2d grid (o spf2,s spf5,^ spf10,D spf20); green P=grasp K-sweep; red*=NIK; orange P=CS")
fig.suptitle("full grasp-pro frontier (K x spokes/frame) vs nik (POST 1px-fix). z15 phantom, vs truth")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_2dgrid_pareto.png", dpi=130); plt.close(fig)
print("SAVED fig_ksweep_pareto.png + fig_2dgrid_pareto.png (corrected NIK overlays)"); print("PARETO_REGEN_DONE")
