"""Visual + perceptual comparison of factorized ranks vs full-rank vs CS.
Static (temporal-mean) images isolate spatial quality. Perceptual vs CS-100 static.
out: rank_visual.png"""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = "/net/beegfs/users/P101440/nik-autoresearch"
sys.path.insert(0, f"{REPO}/baseline_cs"); sys.path.insert(0, f"{REPO}/glue")
import eval as E
from figpath import fig as fpath
D = "/net/beegfs/users/P101440/DCE_NIK"; A = "/net/beegfs/users/P101440/presentation/assets"

def static(path, key=None):
    v = np.abs(np.load(path)); return v.mean(-1)
items = [
    ("full-rank", static(f"{D}/results_nik_verify/nik_slice_13.npy")),
    ("R=5",  static(f"{D}/results_nik_subspace_r5/nik_slice_13.npy")),
    ("R=10", static(f"{D}/results_nik_subspace_r10/nik_slice_13.npy")),
    ("R=20", static(f"{D}/results_nik_subspace_r20/nik_slice_13.npy")),
    ("CS-70", static(f"{A}/arm1_cs70_sl13.npy")),
    ("CS-100", static(f"{A}/arm1_cs100_sl13.npy")),
]
ref = items[-1][1]                                   # CS-100 static = perceptual reference
roi = ref > np.quantile(ref, 0.55)

def pmetrics(img):
    m = E.image_metrics(img, ref, roi)
    return {k: m.get(k, np.nan) for k in ("ssim", "DISTS", "HaarPSI")}

print(f"{'model':10} {'SSIM':>6} {'DISTS':>6} {'HaarPSI':>7}   (static vs CS-100)")
metr = {}
for name, img in items:
    p = pmetrics(img); metr[name] = p
    print(f"{name:10} {p['ssim']:6.3f} {p['DISTS']:6.3f} {p['HaarPSI']:7.3f}")

# figure: full static row + a zoom row (upper-abdomen crop) to expose grain/streak
zc = (slice(40, 130), slice(50, 150))
fig, ax = plt.subplots(2, len(items), figsize=(len(items) * 2.3, 5.2))
for j, (name, img) in enumerate(items):
    vmax = np.percentile(img, 99.5)
    ax[0, j].imshow(np.rot90(img), cmap="gray", vmin=0, vmax=vmax); ax[0, j].set_title(name, fontsize=11)
    ax[1, j].imshow(np.rot90(img[zc]), cmap="gray", vmin=0, vmax=vmax)
    for r in (0, 1): ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
    ax[0, j].set_xlabel(f"HaarPSI {metr[name]['HaarPSI']:.3f}", fontsize=8)
ax[0, 0].set_ylabel("static", fontsize=10); ax[1, 0].set_ylabel("zoom", fontsize=10)
fig.suptitle("Factorized rank vs full-rank vs CS — static image (temporal mean), slice 13",
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(fpath(f"rank_visual.png"), bbox_inches="tight", dpi=150)
print(f"\nwrote {D}/rank_visual.png")
