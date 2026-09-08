"""P0/P3 adjudication: do the temporal-TV or gamma-variate-PK runs beat the BANDLIMIT
FRONTIER on the aorta bolus? Auto-discovers results_pktv_*/nik_slice_13.npy.
Decisive panel = noise vs sharpness scatter: bandlimit runs trace the frontier (lower
t_sigma -> less noise AND less upslope, together); a real temporal win sits BELOW-RIGHT
of that line (higher upslope at equal-or-lower noise). out: aorta_pktv.png"""
import numpy as np, glob, os
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figpath import fig as fpath
D = "/scratch/rnga/vvpshenov/DCE_NIK"; A = "/scratch/rnga/vvpshenov/presentation/assets"
aroi = np.load(f"{D}/aorta_roi.npy"); lroi = np.load(f"{D}/liver_roi.npy")
TA = 374.0

# label -> (style role, color). frontier = base + bandlimit; priors = tv + pk.
STYLE = {
    "base":     ("frontier", "0.45",       "current baseline (t_sigma 1.5)"),
    "bl_ts1p0": ("frontier", "0.62",       "bandlimit t_sigma 1.0"),
    "bl_ts0p7": ("frontier", "0.78",       "bandlimit t_sigma 0.7"),
    "tv0p03":   ("prior",    "tab:green",  "temporal-TV w=0.03"),
    "tv0p1":    ("prior",    "tab:olive",  "temporal-TV w=0.1"),
    "pk_soft":  ("prior",    "tab:purple", "PK gamma (soft residual)"),
    "pk_hard":  ("prior",    "tab:red",    "PK gamma (hard)"),
}
runs = {}
for lab, (role, col, name) in STYLE.items():
    p = f"{D}/results_pktv_{lab}/nik_slice_13.npy"
    if os.path.exists(p): runs[lab] = (np.abs(np.load(p)), role, col, name)
# CS references
CS = {}
for lab, p, col, name in [("cs_fine", f"{A}/arm_temporal_cs100_187.npy", "tab:blue", "CS-100 fine (2s)"),
                          ("cs_coarse", f"{A}/arm1_cs100_sl13.npy", "navy", "CS-100 coarse (31s)")]:
    if os.path.exists(p): CS[lab] = (np.abs(np.load(p)), col, name)

def roi_curve(v, m): return np.array([v[..., i][m].mean() for i in range(v.shape[-1])])
def tvec(nt): return np.linspace(0, TA, nt)
def smooth(c):
    if len(c) < 9: return c
    w = max(5, (len(c) // 12) | 1); return savgol_filter(c, min(w, len(c) - (1 - len(c) % 2)), 3)

def analyze(c, t):
    base = c[t < 55].mean() if (t < 55).any() else c[0]
    sm = smooth(c); fp = t < 200
    peak = sm[fp].max()
    n = (c - base) / (peak - base + 1e-9); ns = (sm - base) / (peak - base + 1e-9)
    up = np.gradient(ns, t)[fp].max()                        # sharpness = max upslope /s
    noise = np.std(c - sm) / (peak - base + 1e-9)            # relative curve noise
    return dict(n=n, up=up, noise=noise)

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
# panel 0: aorta curves ------------------------------------------------------
res = {}
for lab, (vol, role, col, name) in runs.items():
    t = tvec(vol.shape[-1]); r = analyze(roi_curve(vol, aroi), t); res[lab] = (r, role, col, name)
    lw = 2.0 if role == "prior" else 1.3; z = 3 if role == "prior" else 1
    ax[0].plot(t, r["n"], color=col, lw=lw, alpha=.9, zorder=z, label=name)
for lab, (vol, col, name) in CS.items():
    t = tvec(vol.shape[-1]); r = analyze(roi_curve(vol, aroi), t)
    ax[0].plot(t, r["n"], "--", color=col, lw=1.6, alpha=.85, label=name); res[lab] = (r, "cs", col, name)
ax[0].set_title("AORTA bolus (normalized)", fontsize=11); ax[0].set_xlabel("time (s)")
ax[0].set_ylabel("norm. enhancement"); ax[0].legend(fontsize=7.5); ax[0].grid(alpha=.3); ax[0].set_xlim(0, 260)

# panel 1: THE DECISIVE SCATTER  noise (y, lower=better) vs upslope (x, higher=better) ---
for lab, (r, role, col, name) in res.items():
    mk = dict(frontier="s", prior="*", cs="D")[role]; sz = 320 if role == "prior" else 130
    ax[1].scatter(r["up"], r["noise"], marker=mk, s=sz, color=col, edgecolor="k",
                  linewidth=.6, zorder=4 if role == "prior" else 3, label=name)
# draw the bandlimit frontier line through the frontier points
fr = sorted([(res[l][0]["up"], res[l][0]["noise"]) for l in res if res[l][1] == "frontier"])
if len(fr) >= 2:
    fx, fy = zip(*fr); ax[1].plot(fx, fy, "-", color="0.5", lw=1.4, zorder=1)
    ax[1].fill_between(fx, fy, max(fy) * 1.3, color="tab:green", alpha=.05)
ax[1].set_xlabel("sharpness  (max upslope /s)  ->  better"); ax[1].set_ylabel("curve noise  <-  better (lower)")
ax[1].set_title("noise vs sharpness — WIN = below-right of frontier", fontsize=10.5)
ax[1].legend(fontsize=7, loc="upper left"); ax[1].grid(alpha=.3)
ax[1].annotate("bandlimit\nfrontier", (fx[len(fx)//2], fy[len(fy)//2]) if len(fr) >= 2 else (0, 0),
               fontsize=8, color="0.4") if len(fr) >= 2 else None

# panel 2: metric table as bars ----------------------------------------------
labs = list(res); x = np.arange(len(labs)); cols = [res[l][2] for l in labs]
ax[2].bar(x - 0.2, [res[l][0]["noise"] for l in labs], 0.38, color=cols, alpha=.55, label="noise")
ax[2].bar(x + 0.2, [res[l][0]["up"] for l in labs], 0.38, color=cols, alpha=.9, hatch="//", label="upslope")
ax[2].set_xticks(x); ax[2].set_xticklabels([res[l][3] for l in labs], fontsize=6.5, rotation=30, ha="right")
ax[2].set_title("noise (solid) + upslope (hatched)", fontsize=10); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")
fig.suptitle("Aorta bolus — temporal-TV & PK vs the bandlimit frontier (slice 13, R=16)", fontweight="bold")
fig.tight_layout(); fig.savefig(fpath(f"aorta_pktv.png"), bbox_inches="tight", dpi=140)
print(f"{'run':30} {'upslope/s':>10} {'noise':>8}")
for l in res: print(f"{res[l][3]:30} {res[l][0]['up']:10.4f} {res[l][0]['noise']:8.4f}")
print(f"\nfound {len(runs)}/7 pktv runs, {len(CS)} CS refs -> aorta_pktv.png")
