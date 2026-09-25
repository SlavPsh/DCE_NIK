"""painting templates for manual rois: per slice, the model-free frame at 120 s (organs) and 60 s (aorta) as 8-bit png, upscaled 2x (nearest) with the
slice number burnt in the corner. paint pure red (liver), green (spleen), blue (aorta) over a copy and save as results/rois_manual/<ds>_sl<Z>.png.
usage: DCE_DS=p14 python roi_manual_export.py --slices 21,24,27"""
import sys, os, argparse, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); import dsp
ap = argparse.ArgumentParser(); ap.add_argument("--slices", default="21,24,27"); ap.add_argument("--scale", type=int, default=2); a = ap.parse_args()
out = f"/net/beegfs/users/P101440/DCE_NIK/results/rois_manual"; os.makedirs(out, exist_ok=True)
for Z in [int(s) for s in a.slices.split(",")]:
    z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float)
    for ts in (120, 60):
        f = mf[..., (tmf > ts - 10) & (tmf < ts + 10)].mean(2); v = np.clip(f / np.percentile(f, 99.5), 0, 1); im = Image.fromarray((v * 255).astype(np.uint8)).resize((f.shape[1] * a.scale, f.shape[0] * a.scale), Image.NEAREST).convert("RGB")
        ImageDraw.Draw(im).text((6, 6), f"{dsp.DS} slice {Z} t={ts}s  paint: red=liver green=spleen blue=aorta", fill=(255, 255, 0)); im.save(f"{out}/template_{dsp.DS}_sl{Z}_t{ts}.png")
    print("slice", Z, "templates written", f.shape)
