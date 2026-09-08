"""Honest aorta metric: score each recon's aorta curve against a fitted PHYSIOLOGICAL
bolus (baseline + gamma-variate, robust fit). Residual = ALL deviation from a real
bolus shape -> high-freq jitter AND large smooth bumps both count (the short-window
smoother missed the bumps). Sharpness read off the FITTED curve (noise-free upslope).
Also reports washout non-monotonicity (the false-re-enhancement bumps). out: aorta_honest.png"""
import numpy as np, os
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figpath import fig as fpath

def isotonic_decreasing(y):
    """non-increasing least-squares fit via PAVA (pool adjacent violators). No sklearn."""
    y = -np.asarray(y, float)                      # solve non-decreasing on -y
    n = len(y); v = y.copy(); w = np.ones(n); idx = list(range(n + 1))
    lvl = list(range(n)); vals = list(v); wts = list(w); i = 0
    stack = []                                     # (value, weight, count)
    for val in y:
        cw, cc = 1.0, 1
        cv = val
        while stack and stack[-1][0] > cv:
            pv, pw, pc = stack.pop()
            cv = (cv * cw + pv * pw) / (cw + pw); cw += pw; cc += pc
        stack.append((cv, cw, cc))
    out = []
    for cv, cw, cc in stack: out.extend([cv] * cc)
    return -np.asarray(out)
D = "/scratch/rnga/vvpshenov/DCE_NIK"; A = "/scratch/rnga/vvpshenov/presentation/assets"
aroi = np.load(f"{D}/aorta_roi.npy"); TA = 374.0

def gamma(t, base, amp, t0, r, b):
    x = np.clip(t - t0, 0, None)
    return base + amp * (x ** r) * np.exp(-x / b)

def fit_bolus(t, c):
    b0 = float(np.mean(c[t < 40])); pk = float(c.max()); pkt = float(t[np.argmax(c)])
    p0 = [b0, (pk - b0) * 3, max(1.0, pkt - 25), 2.0, 20.0]
    lo = [c.min() - abs(c.min()), 0, 0, 0.3, 3.0]; hi = [pk, (pk - b0) * 5e3 + 1e3, TA, 12, 200]
    try:
        p, _ = curve_fit(gamma, t, c, p0=p0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1 * (pk - b0 + 1e-9), maxfev=20000)
        return gamma(t, *p), p
    except Exception:
        return np.full_like(c, b0), p0

runs = [("base R16", f"{D}/results_pktv_base/nik_slice_13.npy", "0.5"),
        ("bandlimit1.0", f"{D}/results_pktv_bl_ts1p0/nik_slice_13.npy", "0.7"),
        ("bandlimit0.7", f"{D}/results_pktv_bl_ts0p7/nik_slice_13.npy", "tab:orange"),
        ("TV w0.03", f"{D}/results_pktv_tv0p03/nik_slice_13.npy", "tab:green"),
        ("PK soft", f"{D}/results_pktv_pk_soft/nik_slice_13.npy", "tab:purple"),
        ("CS-100 fine", f"{A}/arm_temporal_cs100_187.npy", "tab:blue"),
        ("CS-100 coarse", f"{A}/arm1_cs100_sl13.npy", "navy")]

def acurve(p): v = np.abs(np.load(p)); return np.array([v[..., i][aroi].mean() for i in range(v.shape[-1])])
def tvec(n): return np.linspace(0, TA, n)

res = {}
print(f"{'method':14} {'honest-resid':>12} {'fit-upslope/s':>13} {'washout-bumps':>13}")
for name, p, col in runs:
    if not os.path.exists(p): continue
    c = acurve(p); t = tvec(len(c)); fit, par = fit_bolus(t, c)
    b0 = float(np.mean(c[t < 40])); pk = fit.max(); amp = pk - b0 + 1e-9
    resid = np.sqrt(np.mean((c - fit) ** 2)) / amp                      # ALL-freq deviation from a real bolus
    fp = t < 200; up = np.gradient((fit - b0) / amp, t)[fp].max()        # noise-free sharpness from the fit
    # washout non-monotonicity: after the peak, how much does the curve climb back up?
    pkt = t[np.argmax(fit)]; wt = t > pkt
    if wt.sum() > 5:
        iso = isotonic_decreasing(c[wt])
        bumps = np.sqrt(np.mean((c[wt] - iso) ** 2)) / amp
    else:
        bumps = np.nan
    res[name] = dict(c=c, t=t, fit=fit, resid=resid, up=up, bumps=bumps, col=col)
    print(f"{name:14} {resid:12.4f} {up:13.4f} {bumps:13.4f}")

# figure: curves+fit (top), honest scatter (bottom-left), bars (bottom-right)
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
ax0 = fig.add_subplot(gs[0, :])
for name, r in res.items():
    if "coarse" in name: continue
    n = (r["c"] - r["fit"].min()) / (r["fit"].max() - r["fit"].min() + 1e-9)
    ax0.plot(r["t"], n, color=r["col"], lw=2 if name in ("PK soft", "base R16") else 1.2, alpha=.85, label=name)
ax0.set_xlim(0, 250); ax0.set_title("aorta curves (normalized) — residual from fitted bolus = honest noise", fontsize=11)
ax0.set_xlabel("time (s)"); ax0.set_ylabel("norm. enh."); ax0.legend(fontsize=8, ncol=3); ax0.grid(alpha=.3)
ax1 = fig.add_subplot(gs[1, 0])
for name, r in res.items():
    ax1.scatter(r["up"], r["resid"], s=200, color=r["col"], edgecolor="k", lw=.6, zorder=3)
    ax1.annotate(name, (r["up"], r["resid"]), fontsize=7.5, xytext=(4, 3), textcoords="offset points")
ax1.set_xlabel("sharpness (fitted upslope /s) -> better"); ax1.set_ylabel("honest residual <- better (lower)")
ax1.set_title("HONEST noise vs sharpness — win = lower-right", fontsize=10.5); ax1.grid(alpha=.3)
ax2 = fig.add_subplot(gs[1, 1]); names = list(res); x = np.arange(len(names)); cols = [res[n]["col"] for n in names]
ax2.bar(x - 0.2, [res[n]["resid"] for n in names], 0.38, color=cols, alpha=.6, label="honest residual")
ax2.bar(x + 0.2, [res[n]["bumps"] for n in names], 0.38, color=cols, alpha=.9, hatch="//", label="washout bumps")
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=7, rotation=30, ha="right")
ax2.set_title("residual (solid) + washout non-monotonicity (hatched)", fontsize=10); ax2.legend(fontsize=8); ax2.grid(alpha=.3, axis="y")
fig.suptitle("Aorta — HONEST metric (deviation from a fitted physiological bolus), slice 13", fontweight="bold")
fig.tight_layout(); fig.savefig(fpath(f"aorta_honest.png"), dpi=140, bbox_inches="tight"); print("\nwrote aorta_honest.png")
