"""residual maps vs the all-spoke anatomy (cs100, scaled to each image on the body): what the fine-scale content of each method is, streak texture
or anatomy. full-fov residual at 90 s, kidney zoom of image and residual. same k80 input, one global scale vs model-free.
usage: python residual_fig.py --slice 21 --items "label:path[:key],..." --out figures/residual_sl21.png"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, argparse, numpy as np, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import consolidated as C
from story_figs import ls_scale
TA = 375.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, default=21); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True); ap.add_argument("--t", type=float, default=90.0); a = ap.parse_args(); Z = a.slice
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; body = ctx["BODY"]; cs = ctx["cs100"]; tcs = np.linspace(0, TA, cs.shape[-1])
    z = np.load(f"{B}/step2_slice{Z}.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float)
    def ft(nt): e = np.linspace(0, TA, nt + 1); return 0.5 * (e[:-1] + e[1:])
    def refwin(nt):
        e = np.linspace(0, TA, nt + 1); o = np.zeros(mf.shape[:2] + (nt,), np.float32)
        for g in range(nt):
            m = (tmf >= e[g]) & (tmf < e[g + 1])
            if not m.any(): m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf - 0.5 * (e[g] + e[g + 1])))] = True
            o[:, :, g] = mf[:, :, m].mean(2)
        return o
    def frame(v, t, ts, w=10): m = (t > ts - w) & (t < ts + w); return v[..., m].mean(2) if m.any() else v[..., int(np.argmin(np.abs(t - ts)))]
    r = frame(cs, tcs, a.t); items = []
    for spec in a.items.split(","):
        parts = spec.split(":"); nm, p = parts[0], parts[1]; key = parts[2] if len(parts) > 2 else None
        if not os.path.exists(p): print("missing", p); continue
        zz = np.load(p, allow_pickle=True); v = np.abs(zz[key] if key else zz).astype(np.float32); t = ft(v.shape[-1]); v = ls_scale(v, refwin(v.shape[-1]), body); im = frame(v, t, a.t)
        rr = r * (np.sum(im[body] * r[body]) / (np.sum(r[body] ** 2) + 1e-12)); d = im - rr
        items.append((nm, im, d, float(np.sqrt((d[body] ** 2).mean()) / np.sqrt((rr[body] ** 2).mean())), float(np.sqrt(((d - ndi.gaussian_filter(d, 3.0))[body] ** 2).mean()) / np.sqrt((rr[body] ** 2).mean()))))
    n = len(items); cc = np.argwhere(rois["cortex"]).mean(0).astype(int); h = 44; sl = (slice(max(cc[0] - h, 0), cc[0] + h), slice(max(cc[1] - 2 * h, 0), cc[1] + 2 * h))
    vm = np.percentile(r[body], 99.5); dv = 0.25 * vm
    fig, ax = plt.subplots(3, n, figsize=(3.3 * n, 9.4), gridspec_kw=dict(height_ratios=[1.6, 0.8, 0.8]))
    for j, (nm, im, d, e, ef) in enumerate(items):
        ax[0, j].imshow(d, cmap="RdBu_r", vmin=-dv, vmax=dv); ax[0, j].set_title(nm, fontsize=10, fontweight="bold"); ax[0, j].axis("off")
        ax[0, j].text(0.5, -0.03, f"residual rms {e:.3f}   fine-scale part {ef:.3f}  (of the reference rms)", transform=ax[0, j].transAxes, ha="center", va="top", fontsize=8)
        ax[1, j].imshow(im[sl], cmap="gray", vmin=0, vmax=np.percentile(im[body], 99.5)); ax[1, j].axis("off"); ax[2, j].imshow(d[sl], cmap="RdBu_r", vmin=-dv, vmax=dv); ax[2, j].axis("off")
    ax[1, 0].text(-0.02, 0.5, "kidneys", transform=ax[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9); ax[2, 0].text(-0.02, 0.5, "residual", transform=ax[2, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle(f"slice {Z}, k80, t = {a.t:.0f} s: method minus the grasp-pro all-spoke anatomy (blue low, red high, +-25% of the reference window)", fontsize=11); fig.tight_layout()
    fig.savefig(a.out, dpi=140, facecolor="white"); print("saved", a.out)

if __name__ == "__main__": main()
