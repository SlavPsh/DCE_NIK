"""Score NIK and CS against the method-neutral NUFFT references (slice 13).
  A: all-spokes NUFFT   vs each method's TEMPORAL MEAN
  B: pre-contrast NUFFT vs each method's PRE-CONTRAST MEAN  (+ the clean temporal GT:
     within that window the truth has ZERO trend, so a fitted slope should be 0)
Scale-matched (LS) PSNR + HaarPSI + SSIM on a body ROI. out: figures/score_vs_nufft.png"""
import numpy as np, json, torch, piq, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figpath import fig as fpath
D = "/scratch/rnga/vvpshenov/DCE_NIK"; NUF = f"{D}/results_nufft"
CSD = "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs"
meta = json.load(open(f"{NUF}/meta.json")); TA = 375.0; t_pre = meta["t_pre_s"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

ref_all = np.load(f"{NUF}/nufft_all.npy").astype(np.float32)
ref_pre = np.load(f"{NUF}/nufft_pre.npy").astype(np.float32)
body = ref_all > np.quantile(ref_all, 0.55)

nik = np.abs(np.load(f"{D}/results_spoke_nik_f100/nik_slice_13.npy")).astype(np.float32)
cs = np.abs(np.load(f"{CSD}/cs_slice13_f100.npy")).astype(np.float32)

def win_mean(v, tmax):
    t = np.linspace(0, TA, v.shape[-1])
    m = t < tmax
    return v[..., m].mean(-1), m.sum()

def ls_scale(a, b, m):                     # scale a -> b on ROI (metrics are scale sensitive)
    x, y = a[m], b[m]
    return float((x @ y) / (x @ x + 1e-12))

def tb(img, vmax):
    z = np.clip(img * body / (vmax + 1e-12), 0, 1)
    return torch.from_numpy(z[None, None]).float().to(dev)

def score(pred, ref, tag):
    s = ls_scale(pred, ref, body); p = pred * s
    err = p[body] - ref[body]
    mse = float((err ** 2).mean()); pk = float(ref[body].max())
    psnr = 10 * np.log10(pk ** 2 / (mse + 1e-20))
    vmax = float(np.percentile(ref[body], 99.5))
    with torch.no_grad():
        h = float(piq.haarpsi(tb(p, vmax), tb(ref, vmax), data_range=1.0).cpu())
        ss = float(piq.ssim(tb(p, vmax), tb(ref, vmax), data_range=1.0).cpu())
    print(f"{tag:34} PSNR {psnr:6.2f} dB   HaarPSI {h:.4f}   SSIM {ss:.4f}")
    return dict(psnr=psnr, haarpsi=h, ssim=ss, img=p)

print(f"--- A: all-spokes NUFFT reference vs temporal mean ---")
rA = {"NIK": score(nik.mean(-1), ref_all, "NIK mean vs NUFFT(all)"),
      "CS":  score(cs.mean(-1),  ref_all, "CS  mean vs NUFFT(all)")}
print(f"\n--- B: pre-contrast NUFFT ({meta['n_pre_spokes']} spokes, t<{t_pre:.0f}s) ---")
nik_pre, n_nf = win_mean(nik, t_pre); cs_pre, n_cf = win_mean(cs, t_pre)
print(f"    (NIK {n_nf} frames, CS {n_cf} frames in window)")
rB = {"NIK": score(nik_pre, ref_pre, "NIK pre  vs NUFFT(pre)"),
      "CS":  score(cs_pre,  ref_pre, "CS  pre  vs NUFFT(pre)")}

# --- clean temporal GT: zero trend inside the pre-contrast window ---
print(f"\n--- temporal GT in the pre-contrast window: true trend is ZERO ---")
for name, v in [("NIK", nik), ("CS", cs)]:
    t = np.linspace(0, TA, v.shape[-1]); m = t < t_pre
    c = np.array([v[..., i][body].mean() for i in np.where(m)[0]])
    tt = t[m]; c0 = c / (c.mean() + 1e-12)
    slope = float(np.polyfit(tt, c0, 1)[0]) * 100          # %/s of the window mean
    print(f"  {name:4} drift {slope:+.4f} %/s   (residual std {100*np.std(c0-np.polyval(np.polyfit(tt,c0,1),tt)):.3f} %)")

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for j, (img, t) in enumerate([(ref_all, "NUFFT (all spokes)"), (rA["NIK"]["img"], "NIK mean"), (rA["CS"]["img"], "CS mean")]):
    ax[0, j].imshow(np.rot90(img), cmap="gray", vmin=0, vmax=np.percentile(ref_all, 99.5)); ax[0, j].axis("off"); ax[0, j].set_title(t, fontsize=10)
for j, (img, t) in enumerate([(ref_pre, f"NUFFT (pre, {meta['n_pre_spokes']} spokes)"), (rB["NIK"]["img"], "NIK pre"), (rB["CS"]["img"], "CS pre")]):
    ax[1, j].imshow(np.rot90(img), cmap="gray", vmin=0, vmax=np.percentile(ref_pre, 99.5)); ax[1, j].axis("off"); ax[1, j].set_title(t, fontsize=10)
ax[0, 0].text(-0.08, .5, "A: all spokes", rotation=90, va="center", ha="center", transform=ax[0, 0].transAxes, fontweight="bold")
ax[1, 0].text(-0.08, .5, "B: pre-contrast", rotation=90, va="center", ha="center", transform=ax[1, 0].transAxes, fontweight="bold")
fig.suptitle("NIK and CS vs method-neutral NUFFT references (slice 13)", fontweight="bold")
fig.tight_layout(); fig.savefig(fpath(f"score_vs_nufft.png"), dpi=140, bbox_inches="tight")
print(f"\nwrote figures/score_vs_nufft.png")
