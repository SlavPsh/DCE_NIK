"""export the CS dynamic volume + ROIs for motion inspection across planes/time.
 (1) NIfTI: cs_dynamic_4d (4D X,Y,Z,T), cs_meananat (3D temporal mean), cs_roi_labels (3D
     label map: aorta=1 cortex=2 medulla=3 liver=4 at slices 18/19/21) -> load together in
     fsleyes/ITK-SNAP: scroll axial/coronal/sagittal + time, ROI overlaid.
 (2) GIFs: axial slice-21 over time with ROI contours (in-plane motion); coronal cut through
     the aorta over time with the ROI-slice z-levels marked (through-plane S-I motion).
NOTE voxel size is APPROXIMATE (no header saved): in-plane 3.0mm, slice 4.0mm. adjust if known."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, nibabel as nib, os, sys, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, ".")
import consolidated as C
OUT = "/net/beegfs/users/P101440/DCE_NIK/nifti_export"; os.makedirs(OUT, exist_ok=True)
FIG = "/net/beegfs/users/P101440/DCE_NIK/figures"; TA = 375.0
VOX = (3.0, 3.0, 4.0)                                        # APPROX mm (in-plane, in-plane, slice)
ROI_SLICES = [18, 19, 21]; LAB = {"aorta": 1, "cortex": 2, "medulla": 3, "liver": 4}

vol = np.abs(np.load("/net/beegfs/users/P101440/grasp_pro_py/results_ref_3s/cs_recon_3s.npy")).astype(np.float32)  # [X,Y,Z,T]
X, Y, Z, T = vol.shape; t = np.linspace(0, TA, T)
aff = np.diag([VOX[0], VOX[1], VOX[2], 1.0])

# ---- ROI label volume (from the per-slice segmentation) ----
lab = np.zeros((X, Y, Z), np.int16); ROIS = {}
for zz in ROI_SLICES:
    ctx = C.slice_ctx(zz); ROIS[zz] = ctx["rois"]
    # sanity: cs_recon_3s slice matches the segmentation source
    r = np.corrcoef(vol[:, :, zz, :].mean(-1).ravel(), ctx["cs100"].mean(-1).ravel())[0, 1]
    for nm, v in LAB.items():
        m = ctx["rois"].get(nm)
        if m is not None and m.sum() > 0: lab[:, :, zz][m] = v
    print(f"slice {zz}: ROI voxels aorta/cortex/medulla/liver placed; anat corr vs seg source {r:.3f}", flush=True)

nib.save(nib.Nifti1Image(vol, aff), f"{OUT}/cs_dynamic_4d.nii.gz")
nib.save(nib.Nifti1Image(vol.mean(-1), aff), f"{OUT}/cs_meananat.nii.gz")
nib.save(nib.Nifti1Image(lab, aff), f"{OUT}/cs_roi_labels.nii.gz")
print(f"wrote NIfTI: cs_dynamic_4d {vol.shape}, cs_meananat, cs_roi_labels -> {OUT}", flush=True)

# ---- GIF 1: axial slice 21 over time, ROI contours ----
zz = 21; rois = ROIS[zz]; ax_v = vol[:, :, zz, :]; vmax = np.percentile(ax_v, 99.5)
cols = {"aorta": "#f22", "cortex": "#fa0", "medulla": "#0cf"}
fig, ax = plt.subplots(figsize=(5, 5)); ax.axis("off")
im = ax.imshow(np.rot90(ax_v[..., 0]), cmap="gray", vmax=vmax, animated=True)
for nm, c in cols.items():
    m = rois.get(nm)
    if m is not None and m.sum() > 0: ax.contour(np.rot90(m.astype(float)), levels=[0.5], colors=c, linewidths=1.4)
ttl = ax.set_title("", fontsize=10)
def up_ax(i):
    im.set_data(np.rot90(ax_v[..., i])); ttl.set_text(f"axial slice {zz}  t={t[i]:.0f}s  (ROI contours FIXED)"); return im, ttl
FuncAnimation(fig, up_ax, frames=T, interval=120).save(f"{FIG}/motion_axial_sl21.gif", writer=PillowWriter(fps=8)); plt.close(fig)
print(f"wrote motion_axial_sl21.gif", flush=True)

# ---- GIF 2: coronal cut through the aorta over time (through-plane S-I motion) ----
# cor as [Z, X]: Z (slice / superior-inferior) vertical, X in-plane horizontal. the ROI slices
# are FIXED z-rows; watch anatomy drift up/down through them with breathing.
ay, ax_ = np.where(ROIS[21]["aorta"]); yrow = int(np.round(ay.mean()))            # aorta in-plane row
cor = vol[:, yrow, :, :].transpose(1, 0, 2)                                       # [Z, X, T]
asp = VOX[2] / VOX[0]                                                             # each Z-row is 4mm, X-col 3mm
fig2, a2 = plt.subplots(figsize=(6, 6)); a2.axis("off")
im2 = a2.imshow(cor[..., 0], cmap="gray", vmax=np.percentile(cor, 99.5), aspect=asp, animated=True)
for zz2, c in zip(ROI_SLICES, ["#f22", "#fa0", "#0cf"]):
    a2.axhline(zz2, color=c, ls="--", lw=1.2); a2.text(1, zz2 - 0.3, f"z{zz2}", color=c, fontsize=8)
ttl2 = a2.set_title("", fontsize=10)
def up_cor(i):
    im2.set_data(cor[..., i]); ttl2.set_text(f"coronal thru aorta (y={yrow}) t={t[i]:.0f}s | dashed=fixed ROI z-slices"); return im2, ttl2
FuncAnimation(fig2, up_cor, frames=T, interval=120).save(f"{FIG}/motion_coronal_aorta.gif", writer=PillowWriter(fps=8)); plt.close(fig2)
print(f"wrote motion_coronal_aorta.gif (aorta row y={yrow}, z-levels {ROI_SLICES} marked)", flush=True)
