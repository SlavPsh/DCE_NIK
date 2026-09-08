"""Apples-to-apples: NUFFT vs CS vs NIK(R=16) vs NIK(full-rank) at 100/71/50/29% spokes.
Every method consumes the IDENTICAL acquired spokes (shared keep-files); the aorta mask is
the SAME fixed ROI for all. References are method-neutral full-spoke NUFFT.
  metrics vs NUFFT(all)  on the temporal mean      : PSNR / SSIM / HaarPSI
  metrics vs NUFFT(pre)  on the pre-contrast mean  : PSNR / SSIM / HaarPSI
  held-out consistency   : NUFFT built from the EXCLUDED spokes (data nobody saw)
  aorta bolus            : vs a model-free sliding-window NUFFT reference (all spokes)
out: figures/compare_all_images.png, compare_all_bolus.png, compare_all_metrics.json"""
import numpy as np, finufft, torch, piq, json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
CSD = "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs"
D = "/scratch/rnga/vvpshenov/DCE_NIK"; NUF = f"{D}/results_nufft"
SL = 13; TA = 375.0
meta = json.load(open(f"{NUF}/meta.json")); SIGN = meta["sign"]; t_pre = meta["t_pre_s"]
dev = "cuda" if torch.cuda.is_available() else "cpu"
LABS = [("f100", "100%", 14), ("f70", "71%", 10), ("f50", "50%", 7), ("f25", "29%", 4)]

sh = np.load(f"{REF}/shared.npz")
traj = np.asarray(sh["traj_norm"]).astype(np.complex64)
vt = np.asarray(sh["view_time"]).ravel().astype(np.float64)
nx = int(sh["nx"]); bas = int(sh["bas"])
sl = np.load(f"{REF}/slice_{SL:02d}.npz")
kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64)
b1 = np.asarray(sl["b1"]).astype(np.complex64); ncc = kdata.shape[2]
den = np.sum(np.abs(b1) ** 2, axis=2) + 1e-12
aroi = np.load(f"{D}/aorta_roi.npy")                    # SAME mask for every method
order = np.argsort(vt)

def nufft_img(idx, eps=1e-6):
    tr = traj[:, idx]; w = np.maximum(np.abs(tr), 1.0 / nx / 4.0)
    x = (SIGN * 2 * np.pi * np.real(tr)).astype(np.float64).ravel()
    y = (SIGN * 2 * np.pi * np.imag(tr)).astype(np.float64).ravel()
    acc = np.zeros((nx, nx), dtype=np.complex128)
    for c in range(ncc):
        acc += finufft.nufft2d1(x, y, (kdata[:, idx, c] * w).astype(np.complex128).ravel(),
                                (nx, nx), eps=eps, isign=1) * np.conj(b1[:, :, c])
    im = np.abs(acc / den); s = (nx - bas) // 2
    return im[s:s + bas, s:s + bas].astype(np.float32)

ref_all = np.load(f"{NUF}/nufft_all.npy").astype(np.float32)
ref_pre = np.load(f"{NUF}/nufft_pre.npy").astype(np.float32)
import scipy.ndimage as ndi
# CLEAN body mask: a raw threshold leaks streak specks to the image border, which blows the
# bounding box out to the full width. open -> largest component -> fill holes fixes it.
_raw = ref_all > np.quantile(ref_all, 0.55)
_m0 = ndi.binary_opening(_raw, iterations=2)
_l, _n = ndi.label(_m0)
if _n:
    _m0 = (_l == (1 + int(np.argmax(ndi.sum(np.ones_like(_l), _l, range(1, _n + 1))))))
body = ndi.binary_closing(ndi.binary_fill_holes(_m0), iterations=2)
# PSNR over ROI voxels; HaarPSI/SSIM via per-pixel maps AVERAGED OVER THE MASK ONLY.
# Inputs stay UNMASKED (no artificial boundary edge) and the background never contributes.
from masked_metrics import haarpsi_masked, ssim_masked
from figpath import fig as fpath
MASK_T = torch.from_numpy(body.astype(np.float32))[None, None].to(dev)
print(f"metric ROI: clean body {body.sum()} px ({100*body.mean():.1f}% of frame), hole-free; "
      f"window metrics averaged over mask only", flush=True)

def tb(img, vmax):
    return torch.from_numpy(np.clip(img / (vmax + 1e-12), 0, 1)[None, None]).float().to(dev)

def score(pred, ref):
    s = float((pred[body] @ ref[body]) / (pred[body] @ pred[body] + 1e-12)); p = pred * s
    mse = float(((p[body] - ref[body]) ** 2).mean()); pk = float(ref[body].max())
    psnr = 10 * np.log10(pk ** 2 / (mse + 1e-20)); vmax = float(np.percentile(ref[body], 99.5))
    with torch.no_grad():
        h = float(piq.haarpsi(tb(p, vmax), tb(ref, vmax), data_range=1.0).cpu())
        ss = float(piq.ssim(tb(p, vmax), tb(ref, vmax), data_range=1.0).cpu())
    return dict(psnr=psnr, ssim=ss, haarpsi=h), p

def win_mean(v, tmax):
    t = np.linspace(0, TA, v.shape[-1]); return v[..., t < tmax].mean(-1)

def roi_curve(v):
    t = np.linspace(0, TA, v.shape[-1])
    return t, np.array([v[..., i][aroi].mean() for i in range(v.shape[-1])])

def sliding_curve(idx_pool, W=41, step=10):
    pool = np.array(sorted(idx_pool, key=lambda i: vt[i]))
    ts, cs_ = [], []
    for a in range(0, len(pool) - W + 1, step):
        idx = pool[a:a + W]; ts.append(vt[idx].mean() * TA)
        cs_.append(float(nufft_img(idx, eps=1e-5)[aroi].mean()))
    return np.array(ts), np.array(cs_)

def norm(t, c):
    b = c[t < 50].mean(); pk = c[(t > 50) & (t < 200)].max(); return (c - b) / (pk - b + 1e-9)

def fwhm_ttp(t, n):
    fp = (t > 30) & (t < 200); ab = fp & (n > 0.5)
    return (float(t[fp][np.argmax(n[fp])]),
            float(t[ab].max() - t[ab].min()) if ab.any() else float("nan"))

# ---- model-free bolus reference: ALL spokes (best available estimate of truth) ----
print("model-free bolus reference (all spokes) ...", flush=True)
tr_ref, cr_ref = sliding_curve(np.arange(1708)); nr_ref = norm(tr_ref, cr_ref)
ttp_ref, fw_ref = fwhm_ttp(tr_ref, nr_ref)
print(f"  reference TTP {ttp_ref:.1f}s  FWHM {fw_ref:.1f}s", flush=True)

METHODS = [("NUFFT", None), ("CS", None), ("NIK R=16", None), ("NIK full", None)]
RES, IMGS, BOLUS = {}, {}, {}
for lab, pct, spf in LABS:
    keep = np.load(f"{D}/spoke_masks/keep_{lab}.npy")
    excl = np.setdiff1d(np.arange(1708), keep)
    ho_ref = nufft_img(excl) if len(excl) > 50 else None      # held-out: spokes nobody saw
    paths = {"NUFFT": None,
             "CS": f"{CSD}/cs_slice13_{lab}.npy",
             "NIK R=16": f"{D}/results_spoke_nik_{lab}/nik_slice_13.npy",
             "NIK full": f"{D}/results_spoke_full_{lab}/nik_slice_13.npy"}
    for m, p in paths.items():
        if m == "NUFFT":
            mean_img = np.load(f"{NUF}/frac_{lab}.npy"); pre_img = np.load(f"{NUF}/frac_pre_{lab}.npy")
            tb_, cb_ = sliding_curve(keep)
        else:
            if p is None or not os.path.exists(p):
                print(f"  [{lab}] {m}: MISSING ({p})", flush=True); continue
            v = np.abs(np.load(p)).astype(np.float32)
            mean_img = v.mean(-1); pre_img = win_mean(v, t_pre)
            tb_, cb_ = roi_curve(v)
        sA, pA = score(mean_img, ref_all)
        sB, pB = score(pre_img, ref_pre)
        sH = score(mean_img, ho_ref)[0] if ho_ref is not None else None
        nb = norm(tb_, cb_); ttp, fw = fwhm_ttp(tb_, nb)
        RES[(lab, m)] = dict(pct=pct, spf=spf, A=sA, B=sB, HO=sH, ttp=ttp, fwhm=fw)
        IMGS[(lab, m)] = (pA, pB)
        BOLUS[(lab, m)] = (tb_, nb)
        ho_s = f"{sH['psnr']:6.2f}" if sH else "   n/a"
        print(f"  [{pct:>4}] {m:9} meanPSNR {sA['psnr']:6.2f} Haar {sA['haarpsi']:.3f} | "
              f"prePSNR {sB['psnr']:6.2f} Haar {sB['haarpsi']:.3f} | heldout {ho_s} | "
              f"TTP {ttp:5.1f}s FWHM {fw:6.1f}s", flush=True)

json.dump({f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items()} for k, v in RES.items()},
          open(f"{D}/figures/compare_all_metrics.json", "w"), indent=1, default=float)

# ---------------- images: rows = fraction, cols = ref + methods (mean + pre) -------------
present = [m for m, _ in METHODS if any((l, m) in IMGS for l, _, _ in LABS)]
for which, refimg, ttl in [(0, ref_all, "temporal mean"), (1, ref_pre, "pre-contrast")]:
    ncol = len(present) + 1
    fig, ax = plt.subplots(len(LABS), ncol, figsize=(3.0 * ncol, 3.0 * len(LABS)))
    vm = np.percentile(refimg[body], 99.5)
    for i, (lab, pct, spf) in enumerate(LABS):
        ax[i, 0].imshow(np.rot90(refimg), cmap="gray", vmin=0, vmax=vm); ax[i, 0].axis("off")
        if i == 0: ax[i, 0].set_title("NUFFT reference\n(all spokes)", fontsize=9.5)
        ax[i, 0].text(-0.10, .5, f"{pct}\n({spf} sp/fr)", rotation=90, va="center", ha="center",
                      transform=ax[i, 0].transAxes, fontsize=10, fontweight="bold")
        for j, m in enumerate(present):
            a = ax[i, j + 1]
            if (lab, m) in IMGS:
                a.imshow(np.rot90(IMGS[(lab, m)][which]), cmap="gray", vmin=0, vmax=vm)
                r = RES[(lab, m)]["A" if which == 0 else "B"]
                a.text(.5, .02, f"{r['psnr']:.1f} dB | {r['haarpsi']:.3f}", transform=a.transAxes,
                       ha="center", fontsize=8, color="w")
            a.axis("off")
            if i == 0: a.set_title(m, fontsize=10)
    fig.suptitle(f"NUFFT / CS / NIK at matched spoke fractions — {ttl} (slice 13)\n"
                 f"annotation: PSNR | HaarPSI vs the full-spoke NUFFT reference", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fpath(f"compare_all_{'mean' if which==0 else 'pre'}.png"), dpi=130, bbox_inches="tight")

# ---------------- aorta bolus (same mask everywhere) -------------------------------------
COL = {"NUFFT": "#1d6f4e", "CS": "#0369a1", "NIK R=16": "#7c3aed", "NIK full": "#e0621a"}
fig, ax = plt.subplots(1, len(LABS), figsize=(4.6 * len(LABS), 4.2), sharey=True)
for i, (lab, pct, spf) in enumerate(LABS):
    ax[i].plot(tr_ref, nr_ref, lw=2.4, color="0.25", label="NUFFT ref (all spokes)")
    for m in present:
        if (lab, m) in BOLUS:
            t, n = BOLUS[(lab, m)]; ax[i].plot(t, n, lw=1.5, color=COL[m], alpha=.9, label=m)
    ax[i].axhline(.5, ls=":", color="0.6", lw=1); ax[i].set_xlim(30, 230)
    ax[i].set_title(f"{pct} spokes ({spf} sp/fr)", fontsize=11); ax[i].grid(alpha=.3)
    ax[i].set_xlabel("time (s)")
ax[0].set_ylabel("norm. enhancement"); ax[0].legend(fontsize=8)
fig.suptitle(f"Aorta bolus vs a model-free NUFFT reference (identical ROI mask; ref TTP {ttp_ref:.0f}s, FWHM {fw_ref:.0f}s)",
             fontweight="bold")
fig.tight_layout(); fig.savefig(fpath(f"compare_all_bolus.png"), dpi=135, bbox_inches="tight")
print("\nwrote figures/compare_all_{mean,pre,bolus}.png + compare_all_metrics.json")
