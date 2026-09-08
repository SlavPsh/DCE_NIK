"""Aorta bolus test -- NIK (R=16, support=1.0, continuous-t) vs CS. Same fixed ROIs on all.
CS faces a tradeoff: fine(2s)=noisy OR coarse(31s)=clean-but-smeared. NIK should be BOTH
fine AND clean -- recover the sharp arterial bolus at fine temporal resolution, low noise.
Liver (slow) = counterpoint: all methods should agree. out: aorta_test.png"""
import numpy as np
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/scratch/rnga/vvpshenov/DCE_NIK"; A = "/scratch/rnga/vvpshenov/presentation/assets"
aroi = np.load(f"{D}/aorta_roi.npy"); lroi = np.load(f"{D}/liver_roi.npy")
import os
from figpath import fig as fpath
recons = {}
for tag, path, col, ls in [
    ("NIK R16 (fine)",   f"{D}/results_freq_base/nik_slice_13.npy", "tab:blue", "-"),
    ("NIK R5 (fine)",    f"{D}/results_rebase_r5/nik_slice_13.npy", "tab:cyan", "-"),
    ("CS-100 (2s, fine)",   f"{A}/arm_temporal_cs100_187.npy", "tab:red", "-"),
    ("CS-100 (31s, coarse)",f"{A}/arm1_cs100_sl13.npy", "darkred", "s--")]:
    if os.path.exists(path): recons[tag] = (np.abs(np.load(path)), col, ls)
TA = 374.0
def roi_curve(vol, m): return np.array([vol[..., i][m].mean() for i in range(vol.shape[-1])])
def tvec(nt): return np.linspace(0, TA, nt)
def smooth(c):
    if len(c) < 9: return c
    w = max(5, (len(c)//12)|1); return savgol_filter(c, min(w, len(c)-(1-len(c)%2)), 3)

def analyze(c, t):
    base = c[t < 55].mean() if (t < 55).any() else c[0]
    sm = smooth(c); fp = t < 200                                   # first-pass window
    peak = sm[fp].max(); pk_t = t[fp][np.argmax(sm[fp])]
    plateau = c[t > 300].mean()
    n = (c - base) / (peak - base + 1e-9); ns = (sm - base) / (peak - base + 1e-9)
    up = np.gradient(ns, t)[fp].max()                              # max upslope /s (on smoothed)
    p2p = (peak - base) / (plateau - base + 1e-9)
    half = base + 0.5 * (peak - base); above = fp & (sm > half)    # first-pass FWHM
    fwhm = (t[above].max() - t[above].min()) if above.any() else np.nan
    noise = np.std(c - sm) / (peak - base + 1e-9)                  # relative curve noise
    return dict(n=n, base=base, peak_t=pk_t, up=up, p2p=p2p, fwhm=fwhm, noise=noise)

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))
print(f"{'method':22} {'upslope/s':>9} {'peak/plat':>9} {'FWHM s':>7} {'noise':>7}   (AORTA)")
res = {}
for name, (vol, col, ls) in recons.items():
    t = tvec(vol.shape[-1]); ac = roi_curve(vol, aroi); r = analyze(ac, t); res[name] = r
    ax[0].plot(t, r["n"], ls, color=col, lw=1.6, ms=5, label=name, alpha=.9)
    print(f"{name:22} {r['up']:9.4f} {r['p2p']:9.2f} {r['fwhm']:7.0f} {r['noise']:7.4f}")
ax[0].axhline(0, color="0.8", lw=.7); ax[0].set_title("AORTA — sharp bolus (peak + washout)", fontsize=11)
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("norm. enhancement"); ax[0].legend(fontsize=8.5); ax[0].grid(alpha=.3)

for name, (vol, col, ls) in recons.items():
    t = tvec(vol.shape[-1]); lc = roi_curve(vol, lroi); b = lc[t < 55].mean() if (t<55).any() else lc[0]
    ax[1].plot(t, (lc - b) / (lc.max() - b + 1e-9), ls, color=col, lw=1.6, ms=5, label=name, alpha=.9)
ax[1].set_title("LIVER — slow plateau (counterpoint)", fontsize=11)
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("norm. enhancement"); ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.3)

# metric bars: noise (fine methods) + peak/plateau + upslope
names = list(res); x = np.arange(len(names)); cols = [recons[n][1] for n in names]
ax[2].bar(x - 0.22, [res[n]["noise"] for n in names], 0.2, color=cols, alpha=.55, label="curve noise")
ax[2].bar(x + 0.02, [res[n]["up"]*3 for n in names], 0.2, color=cols, alpha=.85, label="upslope ×3")
ax[2].bar(x + 0.26, [res[n]["p2p"]/6 for n in names], 0.2, color=cols, hatch="//", alpha=.6, label="peak/plat ÷6")
ax[2].set_xticks(x); ax[2].set_xticklabels(names, fontsize=7, rotation=20)
ax[2].set_title("aorta metrics (lower noise + higher sharpness = better)", fontsize=10)
ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")
fig.suptitle("Aorta bolus test — does NIK resolve the sharp bolus cleaner than CS at fine temporal res? (slice 13)", fontweight="bold")
fig.tight_layout(); fig.savefig(fpath(f"aorta_test.png"), bbox_inches="tight", dpi=140)
print("\nwrote aorta_test.png")
