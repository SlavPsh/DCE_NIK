"""side by side of the candidate 'best' pk-arm images vs grasp / grasp-pro / the all-spoke anatomy: full fov at 90 s, kidney zoom at 90 and 300 s,
metric labels (haarpsi / air energy vs cs100, cortex curve nrmse vs model-free). same k80 input, one global scale vs the model-free reference.
usage: python best_image_fig.py --slice 21 --items "label:path[:key],..." --out figures/best_images_sl21.png"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import consolidated as C
from story_figs import ls_scale
TA = 375.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, default=21); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); Z = a.slice
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; body = ctx["BODY"]; air = ctx["AIR"]; cs = ctx["cs100"]; tcs = np.linspace(0, TA, cs.shape[-1])
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
    enh = lambda c, t: c - np.median(c[t < 40]); mfc = np.array([mf[..., i][rois["cortex"]].mean() for i in range(mf.shape[-1])])
    items = []
    for spec in a.items.split(","):
        parts = spec.split(":"); nm, p = parts[0], parts[1]; key = parts[2] if len(parts) > 2 else None
        if nm == "cs100": v, t = cs, tcs
        else:
            if not os.path.exists(p): print("missing", p); continue
            zz = np.load(p, allow_pickle=True); v = np.abs(zz[key] if key else zz).astype(np.float32); t = ft(v.shape[-1]); v = ls_scale(v, refwin(v.shape[-1]), body)
        i90, i300 = frame(v, t, 90), frame(v, t, 300); r90 = frame(cs, tcs, 90)
        s = C.score(i90, r90, body) if nm != "cs100" else dict(haarpsi=1.0); ae = C.bgE(i90, r90, body, air) if nm != "cs100" else C.ref_bgE(r90, body, air)
        c = np.array([v[..., i][rois["cortex"]].mean() for i in range(v.shape[-1])]); ci = np.interp(tmf, t, c); nr = float(np.linalg.norm(enh(ci, tmf) - enh(mfc, tmf)) / np.linalg.norm(enh(mfc, tmf)))
        items.append((nm, i90, i300, s["haarpsi"], ae, nr))
    n = len(items); cc = np.argwhere(rois["cortex"]).mean(0).astype(int); h = 44; sl = (slice(max(cc[0] - h, 0), cc[0] + h), slice(max(cc[1] - 2 * h, 0), cc[1] + 2 * h))
    vmax = np.percentile(frame(cs, tcs, 90)[body], 99.5)
    fig, ax = plt.subplots(3, n, figsize=(3.3 * n, 9.6), gridspec_kw=dict(height_ratios=[1.6, 0.8, 0.8]))
    for j, (nm, i90, i300, hp, ae, nr) in enumerate(items):
        ax[0, j].imshow(i90, cmap="gray", vmin=0, vmax=vmax); ax[0, j].set_title(nm, fontsize=10, fontweight="bold"); ax[0, j].axis("off")
        ax[0, j].text(0.5, -0.03, f"HaarPSI {hp:.3f}   air {ae:.3f}   cortex NRMSE {nr:.3f}", transform=ax[0, j].transAxes, ha="center", va="top", fontsize=8)
        ax[1, j].imshow(i90[sl], cmap="gray", vmin=0, vmax=vmax); ax[1, j].axis("off"); ax[2, j].imshow(i300[sl], cmap="gray", vmin=0, vmax=vmax); ax[2, j].axis("off")
        for k, r in ((1, "cortex"), (1, "medulla")): ax[k, j].contour(rois[r][sl].astype(float), levels=[0.5], colors=["lime" if r == "cortex" else "orange"], linewidths=0.5, alpha=0.7)
    ax[1, 0].text(-0.02, 0.5, "kidneys, 90 s", transform=ax[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9); ax[2, 0].text(-0.02, 0.5, "kidneys, 300 s", transform=ax[2, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.suptitle(f"slice {Z}, k80: image at 90 s (top, metrics vs the grasp-pro all-spoke anatomy and the model-free cortex curve), kidney zoom at 90 and 300 s", fontsize=11); fig.tight_layout()
    fig.savefig(a.out, dpi=140, facecolor="white"); print("saved", a.out)

if __name__ == "__main__": main()
