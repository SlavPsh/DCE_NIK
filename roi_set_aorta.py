"""set the aorta roi of a slice by the blob number from roi_candidates_fig.py (user decision), in the approved-roi file dsp.ROIS(Z).
usage: DCE_DS=p14 python roi_set_aorta.py --slice 21 --blob 3"""
import sys, argparse, numpy as np, scipy.ndimage as ndi
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); import dsp
ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); ap.add_argument("--blob", type=int, default=0); ap.add_argument("--seed", default=None, help="row,col of a point inside the vessel; region grown on the 60 s frame"); ap.add_argument("--radius", type=int, default=16); ap.add_argument("--erode", type=int, default=1); ap.add_argument("--t", type=float, default=60.0); a = ap.parse_args(); Z = a.slice
if a.seed:
    z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float); f = ndi.gaussian_filter(mf[..., (tmf > a.t - 8) & (tmf < a.t + 8)].mean(2), 1.0)
    r0, c0 = [int(v) for v in a.seed.split(",")]; yy, xx = np.ogrid[:f.shape[0], :f.shape[1]]; disc = (yy - r0) ** 2 + (xx - c0) ** 2 <= a.radius ** 2
    loc = f[disc]; thr = 0.5 * (np.percentile(loc, 99) + np.median(loc)); m = disc & (f > thr); lab, n = ndi.label(m)
    m = (lab == lab[r0, c0]) if lab[r0, c0] > 0 else (lab == (1 + int(np.argmax(ndi.sum(np.ones_like(lab), lab, range(1, n + 1)))))); m = ndi.binary_fill_holes(m); tag = f"seed {a.seed}"
else:
    c = np.load(f"{dsp.FIGD}/aorta_candidates{dsp.SFX}_sl{Z}.npz"); lab, ids = c["lab"], c["ids"]; m = ndi.binary_fill_holes(lab == int(ids[a.blob - 1])); tag = f"blob {a.blob}"
if a.erode: m = ndi.binary_erosion(m, iterations=a.erode)
r = dict(np.load(dsp.ROIS(Z), allow_pickle=True)); r["aorta"] = m; r["aorta_blob"] = np.int64(a.blob); r["aorta_tag"] = np.array(tag); np.savez(dsp.ROIS(Z), **r); print(f"slice {Z}: aorta = {tag}, {int(m.sum())} px -> {dsp.ROIS(Z)}")
