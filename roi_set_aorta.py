"""set the aorta roi of a slice by the blob number from roi_candidates_fig.py (user decision), in the approved-roi file dsp.ROIS(Z).
usage: DCE_DS=p14 python roi_set_aorta.py --slice 21 --blob 3"""
import sys, argparse, numpy as np, scipy.ndimage as ndi
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); import dsp
ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); ap.add_argument("--blob", type=int, required=True); ap.add_argument("--erode", type=int, default=1); a = ap.parse_args(); Z = a.slice
c = np.load(f"{dsp.FIGD}/aorta_candidates{dsp.SFX}_sl{Z}.npz"); lab, ids = c["lab"], c["ids"]; m = ndi.binary_fill_holes(lab == int(ids[a.blob - 1]))
if a.erode: m = ndi.binary_erosion(m, iterations=a.erode)
r = dict(np.load(dsp.ROIS(Z), allow_pickle=True)); r["aorta"] = m; r["aorta_blob"] = np.int64(a.blob); np.savez(dsp.ROIS(Z), **r); print(f"slice {Z}: aorta = blob {a.blob}, {int(m.sum())} px -> {dsp.ROIS(Z)}")
