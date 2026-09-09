"""real-data motion gif + clean segmentation overlay, using the NEW both-kidney segmentation
(realkid_slice21.npz) on the model-free dynamic (same grid the masks were segmented on -> aligned).
fixed contours over time -> see anatomy drift under the ROIs (respiratory motion)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.animation import FuncAnimation, PillowWriter
B = "/net/beegfs/users/P101440/DCE_NIK"; OUT = f"{B}/results/realdata_nik_vs_cs_figures/figures"; os.makedirs(OUT, exist_ok=True)
z = np.load(f"{B}/step2_slice21.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0); tmf = z["tmf"]; body = z["body"]
rk = np.load(f"{B}/realkid_slice21.npz"); cortex = np.asarray(rk["cortex"]); medulla = np.asarray(rk["medulla"])
ao = np.asarray(np.load(f"{B}/aif_slice21.npz")["ao"])
masks = {"aorta": ao, "cortex": cortex, "medulla": medulla}; cols = {"aorta": "#f22", "cortex": "#0f0", "medulla": "#fa0"}
vmax = np.percentile(mf[body], 99.5)

# --- clean segmentation overlay (3 phases) ---
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
for a, t in zip(ax, [30, 80, 250]):
    k = int(np.argmin(abs(tmf - t))); a.imshow(mf[:, :, k], cmap="gray", vmax=vmax); a.axis("off"); a.set_title(f"model-free ~{t}s", fontsize=10)
    for nm, c in cols.items():
        if masks[nm].sum(): a.contour(masks[nm], [.5], colors=c, linewidths=1.1)
ax[0].legend(handles=[mlines.Line2D([], [], color=c, label=nm) for nm, c in cols.items()], fontsize=8, loc="lower left")
fig.suptitle("real slice 21: new both-kidney segmentation (green=cortex, orange=medulla, red=aorta)")
fig.tight_layout(); fig.savefig(f"{OUT}/realkid_segmentation.png", dpi=120); plt.close(fig)
print("SAVED realkid_segmentation.png")

# --- motion gif ---
sub = list(range(0, mf.shape[2], 4))                          # subsample time to keep gif light
fig, ax = plt.subplots(figsize=(4, 4), dpi=80); ax.axis("off")
im = ax.imshow(mf[:, :, 0], cmap="gray", vmax=vmax, animated=True)
for nm, c in cols.items():
    if masks[nm].sum(): ax.contour(masks[nm], [.5], colors=c, linewidths=1.0)
ax.legend(handles=[mlines.Line2D([], [], color=c, label=nm) for nm, c in cols.items()], fontsize=7, loc="lower left")
ttl = ax.set_title("", fontsize=9)
def up(i):
    k = sub[i]; im.set_data(mf[:, :, k]); ttl.set_text(f"real model-free slice 21, t={tmf[k]:.0f}s, roi contours fixed"); return im, ttl
FuncAnimation(fig, up, frames=len(sub), interval=130).save(f"{OUT}/fig8_realdata_motion_rois.gif", writer=PillowWriter(fps=8)); plt.close(fig)
print("wrote fig8_realdata_motion_rois.gif  frames %d  size %.1f MB" % (len(sub), os.path.getsize(f"{OUT}/fig8_realdata_motion_rois.gif") / 1e6))
