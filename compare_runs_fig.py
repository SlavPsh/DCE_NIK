"""images and curves for a list of recons on the same k80 input: full image at t_show with metric labels (haarpsi / air energy vs the all-spoke
anatomy), kidney zoom, and roi enhancement curves vs the model-free reference (nrmse in the legend). one global scale vs the reference for all.
usage: python compare_runs_fig.py --slice 21 --items "label:path[:key],..." --out figures/x.png [--t 90]"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import consolidated as C
from story_figs import ls_scale
TA = 375.0; ROIS = ("aorta", "cortex", "medulla")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, default=21); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True); ap.add_argument("--t", type=float, default=90.0); ap.add_argument("--title", default=""); a = ap.parse_args(); Z = a.slice
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
    enh = lambda c, t: c - np.median(c[t < 40]); r90 = frame(cs, tcs, a.t)
    mfc = {r: enh(np.array([mf[..., i][rois[r]].mean() for i in range(mf.shape[-1])]), tmf) for r in ROIS}
    items = []
    for spec in a.items.split(","):
        parts = spec.split(":"); nm, p = parts[0], parts[1]; key = parts[2] if len(parts) > 2 else None
        if not os.path.exists(p): print("missing", p); continue
        zz = np.load(p, allow_pickle=True); v = np.abs(zz[key] if key else zz).astype(np.float32); t = ft(v.shape[-1]); v = ls_scale(v, refwin(v.shape[-1]), body)
        im = frame(v, t, a.t); s = C.score(im, r90, body); ae = C.bgE(im, r90, body, air)
        cur = {r: enh(np.array([v[..., i][rois[r]].mean() for i in range(v.shape[-1])]), t) for r in ROIS}
        nr = {r: float(np.linalg.norm(np.interp(tmf, t, cur[r]) - mfc[r]) / (np.linalg.norm(mfc[r]) + 1e-12)) for r in ROIS}
        items.append(dict(nm=nm, im=im, t=t, cur=cur, nr=nr, hp=s["haarpsi"], ae=ae))
    n = len(items); cc = np.argwhere(rois["cortex"]).mean(0).astype(int); h = 44; sl = (slice(max(cc[0] - h, 0), cc[0] + h), slice(max(cc[1] - 2 * h, 0), cc[1] + 2 * h))
    fig = plt.figure(figsize=(3.4 * n, 11.5)); gs = fig.add_gridspec(3, n, height_ratios=[1.5, 0.75, 1.3], hspace=0.35, wspace=0.08)
    for j, it in enumerate(items):
        ax = fig.add_subplot(gs[0, j]); ax.imshow(it["im"], cmap="gray", vmin=0, vmax=np.percentile(it["im"][body], 99.5)); ax.axis("off"); ax.set_title(it["nm"], fontsize=10, fontweight="bold")
        ax.text(0.5, -0.03, f"HaarPSI {it['hp']:.3f}   air {it['ae']:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax = fig.add_subplot(gs[1, j]); ax.imshow(it["im"][sl], cmap="gray", vmin=0, vmax=np.percentile(it["im"][body], 99.5)); ax.axis("off")
        for r, col in (("cortex", "lime"), ("medulla", "orange")): ax.contour(rois[r][sl].astype(float), levels=[0.5], colors=[col], linewidths=0.5, alpha=0.7)
    cols = plt.cm.tab10(np.linspace(0, 1, 10))
    for k, r in enumerate(ROIS):
        ax = fig.add_subplot(gs[2, k * n // 3:(k + 1) * n // 3] if n >= 3 else gs[2, :]); ax.plot(tmf, mfc[r], "k", lw=2.2, label="model-free")
        for j, it in enumerate(items): ax.plot(it["t"], it["cur"][r], lw=1.2, color=cols[j % 10], label=f"{it['nm']} ({it['nr'][r]:.3f})")
        ax.set_title(f"{r} (curve NRMSE vs model-free in the legend)", fontsize=9); ax.set_xlabel("time (s)"); ax.legend(fontsize=6); ax.axhline(0, color="0.7", lw=0.5)
    fig.suptitle(a.title or f"slice {Z}, k80, image at {a.t:.0f} s: images (metrics vs the grasp-pro all-spoke anatomy), kidney zoom, roi enhancement curves", fontsize=11)
    fig.savefig(a.out, dpi=130, facecolor="white", bbox_inches="tight"); print("saved", a.out)

if __name__ == "__main__": main()
