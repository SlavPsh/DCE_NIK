"""manual rois from painted pngs (results/rois_manual/<ds>_sl<Z>.png, any size = template size): pure red -> liver, green -> spleen, blue -> aorta (tolerant to
anti-aliasing: dominant channel > 150 and the other two < 100). downsampled to the image grid, filled, eroded, written to the approved roi file with the
aorta kept if not painted. then the overlay figure with curves. usage: DCE_DS=p14 python roi_manual_ingest.py --slices 21,24,27"""
import sys, os, argparse, numpy as np, scipy.ndimage as ndi
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B); import dsp
ap = argparse.ArgumentParser(); ap.add_argument("--slices", default="21,24,27"); ap.add_argument("--erode", type=int, default=2); a = ap.parse_args()
for Z in [int(s) for s in a.slices.split(",")]:
    p = f"{B}/results/rois_manual/{dsp.DS}_sl{Z}.png"
    if not os.path.exists(p): print("no painting for slice", Z); continue
    z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float); H, W = mf.shape[:2]
    rgb = np.asarray(Image.open(p).convert("RGB")).astype(int); r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cls = {"liver": (r > 150) & (g < 100) & (b < 100), "spleen": (g > 150) & (r < 100) & (b < 100), "aorta": (b > 150) & (r < 100) & (g < 100)}
    old = dict(np.load(dsp.ROIS(Z), allow_pickle=True)) if os.path.exists(dsp.ROIS(Z)) else {}; rois = {}
    for k, m in cls.items():
        if m.sum() < 20: rois[k] = old[k].astype(bool) if k in old else np.zeros((H, W), bool); print(f"slice {Z} {k}: not painted, kept previous ({int(rois[k].sum())} px)"); continue
        mm = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H), Image.BILINEAR)) > 127; mm = ndi.binary_fill_holes(mm); mm = ndi.binary_erosion(mm, iterations=a.erode) if a.erode else mm; rois[k] = mm
    np.savez(dsp.ROIS(Z), **rois, source=f"manual painting results/rois_manual/{dsp.DS}_sl{Z}.png, eroded {a.erode} px", aorta_tag=np.array("manual")); print("slice", Z, {k: int(v.sum()) for k, v in rois.items()})
    frame = lambda ts, w=10: mf[..., (tmf > ts - w) & (tmf < ts + w)].mean(2); cur = {k: np.array([mf[..., i][v].mean() for i in range(mf.shape[-1])]) for k, v in rois.items() if v.any()}; cur = {k: c - np.median(c[tmf < 40]) for k, c in cur.items()}
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for k, ts in enumerate((60, 120)):
        im = frame(ts); ax[k].imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im, 99.5)); ax[k].set_title(f"model-free at {ts} s", fontsize=10); ax[k].axis("off")
        for nm, c in (("liver", "lime"), ("spleen", "orange"), ("aorta", "cyan")):
            if rois[nm].any(): ax[k].contour(rois[nm].astype(float), levels=[0.5], colors=[c], linewidths=0.9)
    pk = cur["aorta"].max() if "aorta" in cur else 1.0
    for nm, c in (("aorta", "cyan"), ("liver", "lime"), ("spleen", "orange")):
        if nm in cur: ax[2].plot(tmf, cur[nm] / pk, color=c, lw=1.4, label=f"{nm} ({int(rois[nm].sum())} px)")
    ax[2].set_title("model-free mean curves (norm to aorta peak)", fontsize=10); ax[2].set_xlabel("t [s]"); ax[2].legend(fontsize=8); ax[2].axhline(0, color="0.7", lw=0.6)
    fig.suptitle(f"{dsp.DS} slice {Z}: manual rois", fontsize=11); fig.tight_layout(); fig.savefig(f"{dsp.FIGD}/figures/roi_manual{dsp.SFX}_sl{Z}.png", dpi=130, facecolor="white"); plt.close(fig)
print("MANUAL_ROI_DONE")
