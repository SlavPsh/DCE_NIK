"""finer-time CS motion: the GRASP-Pro subspace recon is continuous in t, so re-render at the
native 1.1s grid (cs_img, 3x finer than the 3s volume) and interpolate to 200ms. also a sagittal
view (3s volume) and the 200ms per-spoke respiratory navigator (ground-truth breathing signal).
out GIFs: motion_sagittal_aorta, motion_axial_sl21_fine(1.1s), motion_axial_sl21_200ms; + navigator png."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os, sys, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, ".")
import consolidated as C
from figpath import fig as fpath
REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; FIG = "/net/beegfs/users/P101440/DCE_NIK/figures"; TA = 375.0
VOX = (3.0, 3.0, 4.0)

# ---- (1) SAGITTAL through the aorta, 3s volume ----
vol = np.abs(np.load("/net/beegfs/users/P101440/grasp_pro_py/results_ref_3s/cs_recon_3s.npy")).astype(np.float32)  # [X,Y,Z,T]
X, Y, Z, T = vol.shape; t3 = np.linspace(0, TA, T)
rois21 = C.slice_ctx(21)["rois"]; ax_col, ay_col = np.where(rois21["aorta"])
xcol = int(np.round(ax_col.mean()))                                              # aorta in-plane X (sagittal plane)
sag = vol[xcol, :, :, :].transpose(1, 0, 2)                                       # [Z, Y, T]  (Z vertical=S-I)
asp = VOX[2] / VOX[1]
fig, a = plt.subplots(figsize=(6, 6)); a.axis("off")
im = a.imshow(sag[..., 0], cmap="gray", vmax=np.percentile(sag, 99.5), aspect=asp, animated=True)
for zz, c in zip([18, 19, 21], ["#f22", "#fa0", "#0cf"]): a.axhline(zz, color=c, ls="--", lw=1.2)
ttl = a.set_title("", fontsize=10)
def us(i): im.set_data(sag[..., i]); ttl.set_text(f"sagittal thru aorta (x={xcol}) t={t3[i]:.0f}s | dashed=ROI z-slices"); return im, ttl
FuncAnimation(fig, us, frames=T, interval=120).save(f"{FIG}/motion_sagittal_aorta.gif", writer=PillowWriter(fps=8)); plt.close(fig)
print("wrote motion_sagittal_aorta.gif", flush=True)

# ---- (2) FINE axial slice 21 at native 1.1s (cs_img = CS subspace render, 342 frames) ----
Zk = 21; cs = np.abs(np.load(f"{REF}/slice_{Zk:02d}.npz")["cs_img"]).astype(np.float32)   # [192,192,342]
tf = np.linspace(0, TA, cs.shape[-1]); rois = C.slice_ctx(Zk)["rois"]; vmax = np.percentile(cs, 99.5)
cols = {"aorta": "#f22", "cortex": "#fa0", "medulla": "#0cf"}
def axial_gif(vcube, tt, name, title):
    fig, ax = plt.subplots(figsize=(5, 5)); ax.axis("off")
    im = ax.imshow(np.rot90(vcube[..., 0]), cmap="gray", vmax=vmax, animated=True)
    for nm, c in cols.items():
        m = rois.get(nm)
        if m is not None and m.sum() > 0: ax.contour(np.rot90(m.astype(float)), levels=[0.5], colors=c, linewidths=1.3)
    ti = ax.set_title("", fontsize=10)
    def u(i): im.set_data(np.rot90(vcube[..., i])); ti.set_text(f"{title} t={tt[i]:.1f}s"); return im, ti
    FuncAnimation(fig, u, frames=len(tt), interval=90).save(f"{FIG}/{name}.gif", writer=PillowWriter(fps=11)); plt.close(fig)
    print(f"wrote {name}.gif ({len(tt)} frames, dt~{np.median(np.diff(tt))*1000:.0f}ms)", flush=True)
axial_gif(cs, tf, "motion_axial_sl21_fine", f"CS sl{Zk} native 1.1s + ROIs")
# 200ms: temporally interpolate the smooth subspace render (adds no info below 1.1s, literal request)
t200 = np.arange(0, TA, 0.2); cs200 = np.stack([np.interp(t200, tf, cs[y, x]) for y in range(cs.shape[0]) for x in range(cs.shape[1])], 0).reshape(cs.shape[0], cs.shape[1], -1)
# keep a viewable segment (t 90-160s, kidney enhanced) to bound the gif length
seg = (t200 >= 90) & (t200 <= 160); axial_gif(cs200[..., seg], t200[seg], "motion_axial_sl21_200ms", f"CS sl{Zk} rendered @200ms (interp of 1.1s subspace)")

# ---- (3) 200ms respiratory navigator (per-spoke k-center) ----
sh = np.load(f"{REF}/shared.npz"); vt = np.asarray(sh["view_time"]).ravel(); kd = np.asarray(np.load(f"{REF}/slice_{Zk:02d}.npz")["kdata_radial"])
c0 = kd.shape[0] // 2; nav = np.sqrt((np.abs(kd[c0]) ** 2).sum(-1)); o = np.argsort(vt); nav = nav[o]; ts = vt[o] * TA
from scipy.signal import savgol_filter
navd = nav - savgol_filter(nav, 201, 3)                                           # remove slow enhancement -> motion
fig, ax = plt.subplots(2, 1, figsize=(11, 5))
ax[0].plot(ts, nav / nav.max(), "k", lw=.6); ax[0].set_title(f"200ms k-center navigator, slice {Zk} (dt~{np.median(np.diff(ts))*1000:.0f}ms) - full"); ax[0].set_xlabel("s")
seg2 = (ts > 100) & (ts < 140); ax[1].plot(ts[seg2], navd[seg2], "b", lw=.8); ax[1].set_title("detrended, 100-140s zoom -> respiratory + cardiac oscillation"); ax[1].set_xlabel("s"); ax[1].grid(alpha=.3)
fig.tight_layout(); p = fpath("motion_navigator_200ms.png"); fig.savefig(p, dpi=130); plt.close(fig); print(f"wrote {p.split('/')[-1]}", flush=True)
