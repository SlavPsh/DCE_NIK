"""Aggregate the physical no-motion comparison: ALL methods (NIK-F0 / NIK-subspace-R5 / NIK-free /
GRASP-Pro) side-by-side vs XCAT truth. Metrics next to pictures: montage, error maps, ROI curves,
metric-vs-time, animation. Body-masked NRMSE/SSIM/PSNR/HaarPSI + curves(peak-time/FWHM). Seeds ->
mean+-sd; a representative (median-NRMSE) seed drives the figures. usage: python xph_aggregate.py"""
import warnings; warnings.filterwarnings("ignore")
import os, glob, json, csv, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.ndimage import uniform_filter
import xph_pipeline as P, xph_common as X
OUT = X.OUT; FIG = f"{OUT}/figures"; ANI = f"{OUT}/animations"; A = f"{OUT}/arrays"
import os as _os
# phantom reference-method plumbing. defaults = grasp-pro (unchanged). grasp v2:
#   GRASP_NPZ=grasp_v2_recon.npz GRASP_LABEL=GRASP-v2 TAG=_gv2
_GNPZ = _os.environ.get("GRASP_NPZ", "grasp_recon.npz")
_GLAB = _os.environ.get("GRASP_LABEL", "GRASP-Pro")
_TAG = _os.environ.get("TAG", "")

sel = json.load(open(f"{A}/stageA_selection.json")); W = sel["selected_width"]
KS = json.load(open(f"{A}/stageB_selection.json"))["selected_ks"] if os.path.exists(f"{A}/stageB_selection.json") else 2.5
d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
rv = float(Tr[body].max() - Tr[body].min()); _p99 = np.percentile(Tr[body], 99) + 1e-9
PH = dict(precontrast=(0, 18), first_pass=(18, 45), cortical=(45, 90), late=(90, 200))
def mnr(a, b): return float(np.sqrt(np.mean((a[body]-b[body])**2))/(rv+1e-12))
def ssim(a, b, win=7):
    C1 = (0.01*rv)**2; C2 = (0.03*rv)**2; ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    va = uniform_filter(a*a, win)-ma**2; vb = uniform_filter(b*b, win)-mb**2; vab = uniform_filter(a*b, win)-ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2)))[body].mean())
def psnr(a, b): return float(20*np.log10(rv/(np.sqrt(np.mean((a[body]-b[body])**2))+1e-12)))
import torch as _t, piq
_ys, _xs = np.where(body); _y0, _y1, _x0, _x1 = _ys.min(), _ys.max()+1, _xs.min(), _xs.max()+1   # anatomy bbox
def haar(a, b):  # standard HaarPSI [0,1] over the anatomy bbox (higher=better)
    xa = _t.tensor(np.clip(a[_y0:_y1, _x0:_x1]/_p99, 0, 1), dtype=_t.float32)[None, None]
    xb = _t.tensor(np.clip(b[_y0:_y1, _x0:_x1]/_p99, 0, 1), dtype=_t.float32)[None, None]
    try: return float(piq.haarpsi(xa, xb, data_range=1.0).item())
    except Exception: return float("nan")

# ---- collect all methods: {name: {seed: rec_dynamic}} ----
M = {}
for f in sorted(glob.glob(f"{A}/nik_eval_w{W}_ks{KS:g}_s*.npz")):
    s = int(f.split("_s")[-1].split(".")[0]); M.setdefault("NIK-F0", {})[s] = np.load(f)["rec_best"]
import re as _re
for f in sorted(glob.glob(f"{A}/img_eval_sub*_w{W}_s*.npz")):
    Rk = _re.search(r"sub(\d+)_w", f).group(1); s = int(f.split("_s")[-1].split(".")[0])
    M.setdefault(f"NIK-sub{Rk}", {})[s] = np.load(f)["rec_best"]
for f in sorted(glob.glob(f"{A}/img_eval_free_w{W}_s*.npz")):
    s = int(f.split("_s")[-1].split(".")[0]); M.setdefault("NIK-free", {})[s] = np.load(f)["rec_best"]
if os.path.exists(f"{A}/{_GNPZ}"): M[_GLAB] = {0: np.load(f"{A}/{_GNPZ}")["rec"]}
# in-file reference recon (the sim's own reconstruction; NOT ground truth) at same 344-frame timing
try:
    import h5py as _h5
    _rf = _h5.File(X.SIM, "r"); _rc = np.abs(np.array(_rf["results"]["images"]["Recon"]["img"])[:, P.ZI]).astype(np.float32)
    _cs = np.stack([X._embed(_rc[i], _rc.shape[1]) for i in range(_rc.shape[0])], -1)   # (RO,RO,F)
    def _cc(a, b): a = a.ravel()-a.mean(); b = b.ravel()-b.mean(); return float((a*b).sum()/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
    if _cc(_cs.mean(2), Tr.mean(2)) < _cc(_cs[::-1, ::-1].mean(2), Tr.mean(2)): _cs = _cs[::-1, ::-1]   # match truth orientation
    if _cs.shape[2] == len(tq): M["CS-file"] = {0: _cs}
    else: print("CS-file frame mismatch:", _cs.shape[2], "vs", len(tq))
except Exception as e: print("CS-file skipped:", str(e)[:120])
# uniform global LS scale-to-truth per method/seed (fair magnitude; ~1.0 for already-scaled NIK/GRASP, corrects in-file CS ~1.73x)
for m in M:
    for s in list(M[m]):
        v = M[m][s]; sc = float((v[body] * Tr[body]).sum() / ((v[body] ** 2).sum() + 1e-9)); M[m][s] = v * sc
ORDER = [m for m in ["NIK-F0", "NIK-sub5", "NIK-sub12", "NIK-sub16", "NIK-free", _GLAB, "CS-file"] if m in M]
print("methods:", {m: sorted(M[m]) for m in ORDER})

# per-method per-frame metrics for every seed; representative seed = median overall NRMSE
def per_frame(rec, fn): return np.array([fn(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])
seed_nrmse = {m: {s: per_frame(M[m][s], mnr).mean() for s in M[m]} for m in ORDER}
rep = {m: sorted(M[m], key=lambda s: abs(seed_nrmse[m][s]-np.median(list(seed_nrmse[m].values()))))[0] for m in ORDER}
recM = {m: M[m][rep[m]] for m in ORDER}
NR = {m: per_frame(recM[m], mnr) for m in ORDER}; SS = {m: per_frame(recM[m], ssim) for m in ORDER}
PS = {m: per_frame(recM[m], psnr) for m in ORDER}; HP = {m: per_frame(recM[m], haar) for m in ORDER}
def phase(a): return {k: float(a[(tq >= lo) & (tq < hi)].mean()) for k, (lo, hi) in PH.items()}

# ---- CSVs ----
with open(f"{OUT}/image_metrics_per_frame{_TAG}.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["frame", "time_s"] + [f"{m}_{q}" for m in ORDER for q in ("NRMSE", "SSIM", "PSNR", "HaarPSI")])
    for t in range(len(tq)): w.writerow([t, f"{tq[t]:.2f}"] + [f"{v[m][t]:.4f}" for m in ORDER for v in (NR, SS, PS, HP)])
with open(f"{OUT}/image_metrics_summary{_TAG}.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["metric"] + ORDER)
    w.writerow(["NRMSE_overall(mean+-sd)"] + [f"{np.mean(list(seed_nrmse[m].values())):.4f}+-{np.std(list(seed_nrmse[m].values())):.4f}" for m in ORDER])
    for lab, v in [("SSIM_overall", SS), ("PSNR_overall_dB", PS), ("HaarPSI_overall", HP)]:
        w.writerow([lab] + [f"{np.nanmean(v[m]):.4f}" for m in ORDER])
    for q, v in [("NRMSE", NR), ("SSIM", SS), ("HaarPSI", HP)]:
        for k in PH: w.writerow([f"{q}_{k}"] + [f"{phase(v[m])[k]:.4f}" for m in ORDER])
# curves + first-pass
def fp_metrics(c):
    b = c[:20].mean(); n = c - b; pk = n.max(); ttp = tq[n.argmax()]; idx = np.where((n >= pk/2) & (tq < ttp+50))[0]
    return ttp, (float(tq[idx[-1]]-tq[idx[0]]) if idx.size > 1 else 0.0), float(c.max())
_pre = tq < 18                                               # precontrast frames
def _bsub(c): return c - np.median(c[_pre])                  # global baseline subtraction, one scalar all frames -> enhancement
curves = {m: {nm: _bsub(np.median(recM[m][R[nm]], 0)) for nm in ["aorta", "cortex", "medulla"]} for m in ORDER}   # median + baseline-subtracted
truec = {nm: _bsub(np.median(Tr[R[nm]], 0)) for nm in ["aorta", "cortex", "medulla"]}
with open(f"{OUT}/curve_metrics{_TAG}.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["ROI/quantity", "truth"] + ORDER)
    for nm in ["aorta", "cortex", "medulla"]:
        w.writerow([f"{nm}_curve_NRMSE(mean+-sd)", "-"] + [f"{np.mean([np.linalg.norm(_bsub(np.median(M[m][s][R[nm]], 0))-truec[nm])/np.linalg.norm(truec[nm]) for s in M[m]]):.4f}+-{np.std([np.linalg.norm(_bsub(np.median(M[m][s][R[nm]], 0))-truec[nm])/np.linalg.norm(truec[nm]) for s in M[m]]):.4f}" for m in ORDER])
    tt = fp_metrics(truec["aorta"])
    w.writerow(["aorta_peak_time_s", f"{tt[0]:.1f}"] + [f"{fp_metrics(curves[m]['aorta'])[0]:.1f}" for m in ORDER])
    w.writerow(["aorta_FWHM_s", f"{tt[1]:.1f}"] + [f"{fp_metrics(curves[m]['aorta'])[1]:.1f}" for m in ORDER])
    w.writerow(["aorta_peak_amp", f"{tt[2]:.3f}"] + [f"{fp_metrics(curves[m]['aorta'])[2]:.3f}" for m in ORDER])
# kspace (NIK variants have test spokes; GRASP frame-based -> n/a)
with open(f"{OUT}/kspace_metrics{_TAG}.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["method", "test_NMSE(mean+-sd)", "test_outer_k", "note"])
    for m, pre in [("NIK-F0", f"nik_eval_w{W}_ks{KS:g}"), ("NIK-sub5", f"img_eval_sub5_w{W}"), ("NIK-free", f"img_eval_free_w{W}")]:
        fs = sorted(glob.glob(f"{A}/{pre}_s*.npz"))
        if not fs: continue
        tn = [float(np.load(f)["test_best"][0]) for f in fs]; to = [float(np.load(f)["test_best"][3]) for f in fs]
        w.writerow([m, f"{np.mean(tn):.3e}+-{np.std(tn):.0e}", f"{np.mean(to):.3e}", "direct model k-query on untouched test spokes"])
    w.writerow([_GLAB, "n/a", "n/a", "frame-based: no comparable held-out spoke prediction"])

# ================= FIGURES (metrics next to pictures) =================
os.makedirs(FIG, exist_ok=True); os.makedirs(ANI, exist_ok=True)
frames = [np.argmin(abs(tq-t)) for t in (10, 28, 39, 90, 160)]; vmax = _p99
rows = ["truth"] + ORDER
# Fig0 ROI-overlay VERIFICATION: aorta/cortex/medulla contours on truth anatomy (confirm placement before trusting curves)
_rc = [("aorta", "red"), ("cortex", "lime"), ("medulla", "orange")]
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
for a, t, ttl in zip(ax, [np.argmin(abs(tq-28)), np.argmin(abs(tq-90)), np.argmin(abs(tq-160))], ["first-pass ~28s", "cortical ~90s", "late ~160s"]):
    a.imshow(Tr[:, :, t], cmap="gray", vmax=vmax); a.set_xticks([]); a.set_yticks([]); a.set_title(f"truth {ttl}", fontsize=9)
    for nm, col in _rc:
        if R[nm].sum(): a.contour(R[nm], levels=[0.5], colors=col, linewidths=1.2)
_lg = [plt.Line2D([0], [0], color=c, lw=2, label=f"{nm} (n={int(R[nm].sum())})") for nm, c in _rc]
ax[0].legend(handles=_lg, fontsize=7, loc="lower left")
fig.suptitle(f"roi placement, truth slice z{P.ZI}. red aorta, green cortex, orange medulla"); fig.tight_layout()
fig.savefig(f"{FIG}/fig0_roi_check{_TAG}.png", dpi=120); plt.close(fig)
# Fig1 montage: rows = truth + methods, cols = key DCE phases
fig, ax = plt.subplots(len(rows), len(frames), figsize=(2.2*len(frames), 2.1*len(rows)), squeeze=False)
for j, t in enumerate(frames):
    ax[0, j].imshow(Tr[:, :, t], cmap="gray", vmax=vmax); ax[0, j].set_title(f"{tq[t]:.0f}s", fontsize=8)
    for r, m in enumerate(ORDER, 1): ax[r, j].imshow(recM[m][:, :, t], cmap="gray", vmax=vmax)
for r, lab in enumerate(rows): ax[r, 0].set_ylabel(lab, fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.suptitle("recons. truth top, methods below, dce phases"); fig.tight_layout(); fig.savefig(f"{FIG}/fig1_montage{_TAG}.png", dpi=120); plt.close(fig)
# Fig2 error maps
fig, ax = plt.subplots(len(ORDER), len(frames), figsize=(2.2*len(frames), 2.1*len(ORDER)), squeeze=False)
for r, m in enumerate(ORDER):
    for j, t in enumerate(frames): ax[r, j].imshow(np.abs(recM[m][:, :, t]-Tr[:, :, t]), cmap="magma", vmax=vmax*0.5); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
    ax[r, 0].set_ylabel(m, fontsize=9)
for j, t in enumerate(frames): ax[0, j].set_title(f"{tq[t]:.0f}s", fontsize=8)
fig.suptitle("error maps, recon vs truth"); fig.tight_layout(); fig.savefig(f"{FIG}/fig2_errormaps{_TAG}.png", dpi=120); plt.close(fig)
# Fig3 ROI curves (all methods vs truth) + metric annotation
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for k, nm in enumerate(["aorta", "cortex", "medulla"]):
    ax[k].plot(tq, truec[nm], "k-", lw=2.5, label="truth")
    for m in ORDER: ax[k].plot(tq, curves[m][nm], lw=1.2, label=m)
    ax[k].set_title(f"{nm}"); ax[k].set_xlabel("time (s)")
ax[0].legend(fontsize=7); fig.suptitle("roi enhancement curves, baseline-subtracted"); fig.tight_layout(); fig.savefig(f"{FIG}/fig3_roi_curves{_TAG}.png", dpi=120); plt.close(fig)
# Fig4 metric vs time (all 4 metrics x all methods)
fig, ax = plt.subplots(2, 2, figsize=(12, 7))
for a, (lab, v) in zip(ax.ravel(), [("masked NRMSE", NR), ("SSIM", SS), ("PSNR dB", PS), ("HaarPSI", HP)]):
    for m in ORDER: a.plot(tq, v[m], lw=1, label=m)
    a.set_title(lab); a.set_xlabel("s"); [a.axvspan(lo, hi, alpha=0.05, color="k") for lo, hi in PH.values()]
ax[0, 0].legend(fontsize=7); fig.suptitle("image metrics vs time, dce phases shaded"); fig.tight_layout(); fig.savefig(f"{FIG}/fig4_metric_vs_time{_TAG}.png", dpi=120); plt.close(fig)
# Fig5 hyperparameter diagnostics
allc = {}
for f in sorted(glob.glob(f"{A}/nik_eval_w*_ks*_s*.npz")):
    e = np.load(f); allc[(int(e["width"]), float(e["k_sigma"]), int(e["seed"]))] = e
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for (wd, kk, s), e in allc.items():
    if kk == KS and s == 0: ax[0].scatter(wd, float(e["img_nrmse_mean_best"])); ax[0].annotate(f"w{wd}", (wd, float(e["img_nrmse_mean_best"])), fontsize=6)
    if wd == W and s == 0: ax[1].scatter(kk, float(e["img_nrmse_mean_best"])); ax[1].annotate(f"ks{kk:g}", (kk, float(e["img_nrmse_mean_best"])), fontsize=6)
ax[0].set_xlabel("width"); ax[0].set_ylabel("truth img NRMSE (F0)"); ax[0].set_title("capacity vs truth")
ax[1].set_xlabel("k_sigma"); ax[1].set_title("spatial-freq vs truth"); fig.tight_layout(); fig.savefig(f"{FIG}/fig5_hyperparam_diag{_TAG}.png", dpi=110); plt.close(fig)

# ---- animation: truth + all methods synchronized ----
sub = list(range(0, len(tq), 3)); fig, ax = plt.subplots(1, len(rows), figsize=(2.1*len(rows), 2.6))
for a, lab in zip(ax, rows):
    a.axis("off"); a.set_title(lab, fontsize=8)
    for nm, col in _rc:                                  # ROI contours on every panel (verify placement over time)
        if R[nm].sum(): a.contour(R[nm], levels=[0.5], colors=col, linewidths=0.7)
def fr(i):
    t = sub[i]
    for a in ax: [im.remove() for im in list(a.images)]
    ax[0].imshow(Tr[:, :, t], cmap="gray", vmax=vmax)
    for r, m in enumerate(ORDER, 1): ax[r].imshow(recM[m][:, :, t], cmap="gray", vmax=vmax)
    fig.suptitle(f"t = {tq[t]:.1f} s", fontsize=9); return []
an = animation.FuncAnimation(fig, fr, frames=len(sub), interval=120)
try: an.save(f"{ANI}/truth_vs_methods.mp4", dpi=90, writer="ffmpeg")
except Exception: an.save(f"{ANI}/truth_vs_methods.gif", dpi=70, writer="pillow")
plt.close(fig)

# ================= PK maps (signal-domain Patlak; RELATIVE, no signal->[Gd] conversion) =================
try:
    _aif = np.load("aif_xph.npz"); aiff = _aif["aif_frame"].astype(np.float64); integ = _aif["integ"].astype(np.float64)
    Dsg = np.stack([integ, aiff], 1); Dpinv = np.linalg.pinv(Dsg)        # (F,2)=[Ktrans-basis, vp-basis]
    pre = tq < 18                                                        # precontrast baseline
    def patlak(V):
        base = V[:, :, pre].mean(2); enh = (V - base[:, :, None]).reshape(-1, len(tq))
        Pm = Dpinv @ enh.T                                              # (2,npix)
        return Pm[0].reshape(V.shape[:2]) * body, Pm[1].reshape(V.shape[:2]) * body
    PKt = {"truth": patlak(Tr)}; PKt.update({m: patlak(recM[m]) for m in ORDER}); pkrows = ["truth"] + ORDER
    kmax = np.percentile(PKt["truth"][0][body], 98); vmx = np.percentile(PKt["truth"][1][body], 98)
    fig, ax = plt.subplots(2, len(pkrows), figsize=(2.1*len(pkrows), 4.4), squeeze=False)
    for c, m in enumerate(pkrows):
        ax[0, c].imshow(PKt[m][0], cmap="inferno", vmin=0, vmax=kmax); ax[0, c].set_title(m, fontsize=8)
        ax[1, c].imshow(PKt[m][1], cmap="viridis", vmin=0, vmax=vmx)
        for rr in (0, 1): ax[rr, c].set_xticks([]); ax[rr, c].set_yticks([])
    ax[0, 0].set_ylabel("Ktrans-analog", fontsize=9); ax[1, 0].set_ylabel("vp-analog", fontsize=9)
    fig.suptitle("signal-domain patlak maps, relative"); fig.tight_layout()
    fig.savefig(f"{FIG}/fig6_pk_maps{_TAG}.png", dpi=120); plt.close(fig)
    with open(f"{OUT}/pk_metrics{_TAG}.csv", "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["param/ROI (signal-Patlak)", "truth"] + ORDER)
        for pi, pn in [(0, "Ktrans"), (1, "vp")]:
            for nm in ["aorta", "cortex", "medulla"]:
                w.writerow([f"{pn}_{nm}", f"{np.median(PKt['truth'][pi][R[nm]]):.4f}"] + [f"{np.median(PKt[m][pi][R[nm]]):.4f}" for m in ORDER])
    print("PK maps (fig6) + pk_metrics.csv done")
except Exception as e:
    print("PK section skipped:", str(e)[:140])

print("\n==== SUMMARY (physical no-motion, vs XCAT truth) ====")
print(f"{'method':10} {'NRMSE':16} {'SSIM':7} {'PSNR':7} {'HaarPSI':8} {'aortaCurve':11} {'cortex':7} {'medulla':7}")
for m in ORDER:
    cn = {nm: np.mean([np.linalg.norm(_bsub(np.median(M[m][s][R[nm]], 0))-truec[nm])/np.linalg.norm(truec[nm]) for s in M[m]]) for nm in ["aorta", "cortex", "medulla"]}
    print(f"{m:10} {np.mean(list(seed_nrmse[m].values())):.4f}+-{np.std(list(seed_nrmse[m].values())):.4f}   {np.nanmean(SS[m]):.3f}  {np.nanmean(PS[m]):5.1f}  {np.nanmean(HP[m]):.3f}   {cn['aorta']:.4f}     {cn['cortex']:.3f}  {cn['medulla']:.3f}")
print("wrote CSVs + figures(1-5) + animation ->", OUT); print("AGG_DONE")
