"""side-by-side dynamic gif of the production recons of one slice: every arm resampled to a common time grid (window mean, +-w s), scaled once
to the model-free late anatomy (one global scale per arm, no per-frame normalization), one grey window for all, roi contours optional, time stamp
and the roi curves of the shown arms below (dot = current time). input consistency is asserted from the files: every arm must have the dataset's
grid, and the caption states the spoke set (k80, v%10<8) which the recon scripts enforce in code (train_grasp_nik --spoke-keep-file,
grasp_v2_real keep_mask, cs_nikmatch keep_mask).
usage: DCE_DS=p14 python recon_gif.py --slice 24 --items "label:path,..." --out <gif> [--dt 4 --w 6 --fps 6 --rois 1]"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp, consolidated as C

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=4.0); ap.add_argument("--w", type=float, default=6.0); ap.add_argument("--fps", type=int, default=6); ap.add_argument("--rois", type=int, default=1); ap.add_argument("--note", default=""); ap.add_argument("--single", type=int, default=0, help="1 = one plain gif per arm into --out (a directory)"); ap.add_argument("--single-tag", default=""); ap.add_argument("--t0", type=float, default=None); ap.add_argument("--t1", type=float, default=None)
    a = ap.parse_args(); Z = a.slice; TA = dsp.TA; ctx = C.slice_ctx(Z); body = ctx["BODY"]; rois = ctx["rois"]
    ref = np.load(f"{dsp.NUF(Z)}/nufft_late.npy").astype(np.float32)                                  # model-free late anatomy: one scale per arm
    z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float)
    tg = np.arange(a.t0 if a.t0 is not None else a.w, (a.t1 if a.t1 is not None else TA - a.w) + 1e-6, a.dt); arms = []    # --t0 / --t1: a time excerpt (the arrival window)
    def resample(v, t):
        o = np.empty(body.shape + (len(tg),), np.float32)
        for i, ts in enumerate(tg):
            m = (t > ts - a.w) & (t < ts + a.w); o[..., i] = v[..., m].mean(2) if m.any() else v[..., int(np.argmin(np.abs(t - ts)))]
        return o
    for spec in [x for x in a.items.split(",") if x]:
        lab, path = spec.split(":", 1)
        if not os.path.exists(path): print("missing", path); continue
        v = np.abs(np.load(path)).astype(np.float32); assert v.shape[:2] == body.shape, (lab, v.shape, body.shape)
        t = (np.arange(v.shape[-1]) + 0.5) * TA / v.shape[-1]; late = v[..., t > 200].mean(2); s = np.sum(late[body] * ref[body]) / (np.sum(late[body] ** 2) + 1e-12)
        arms.append((lab, resample(v * s, t), v.shape[-1]))
    s = np.sum(mf[..., tmf > 200].mean(2)[body] * ref[body]) / (np.sum(mf[..., tmf > 200].mean(2)[body] ** 2) + 1e-12); arms.insert(0, ("model-free 31-spoke", resample(mf * s, tmf), mf.shape[-1]))
    vmax = np.percentile(ref[body], 99.5) * 1.15; names = [r for r in dsp.ROI_NAMES if r in rois]; cols = dict(zip(names, ("cyan", "lime", "orange")))
    curves = {lab: {r: np.array([vv[..., i][rois[r]].mean() for i in range(len(tg))]) for r in names} for lab, vv, _ in arms}
    if a.single:                                                                                       # one plain gif per arm: image, time stamp, 2x upscale, no rois, no curves
        from PIL import Image, ImageDraw
        os.makedirs(a.out, exist_ok=True); tag = "" if not a.single_tag else "_" + a.single_tag
        for lab, vv, nt in arms:
            fn = lab.split("(")[0].strip().replace(" ", "_").replace("+", "_").replace("-", "").lower(); frames = []
            for i in range(len(tg)):
                im = np.clip(vv[..., i] / vmax, 0, 1) * 255; pil = Image.fromarray(im.astype(np.uint8)).resize((vv.shape[1] * 2, vv.shape[0] * 2), Image.LANCZOS)
                ImageDraw.Draw(pil).text((8, 6), f"{lab}   t = {tg[i]:4.0f} s", fill=255); frames.append(pil)
            out = f"{a.out}/{dsp.DS}_sl{Z}_{fn}{tag}.gif"; frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000 // a.fps, loop=0, optimize=False); print("saved", out, f"{os.path.getsize(out) / 1e6:.1f} MB")
        print("GIF_DONE"); return
    n = len(arms); fig = plt.figure(figsize=(2.9 * n, 6.2)); gs = fig.add_gridspec(2, n, height_ratios=[1.0, 0.55])
    axs = [fig.add_subplot(gs[0, j]) for j in range(n)]; hs = []
    for ax, (lab, vv, nt) in zip(axs, arms):
        hs.append(ax.imshow(vv[..., 0], cmap="gray", vmin=0, vmax=vmax, animated=True)); ax.set_title(f"{lab}\n({nt} frames)", fontsize=8); ax.axis("off")
        if a.rois:
            for r in names: ax.contour(rois[r].astype(float), levels=[0.5], colors=[cols[r]], linewidths=0.5)
    cax = [fig.add_subplot(gs[1, j]) for j in range(n)]; dots = []
    for ax, (lab, _, _) in zip(cax, arms):
        for r in names: ax.plot(tg, curves[lab][r], color=cols[r], lw=0.9, label=r)
        ax.set_ylim(0, max(curves[l][r].max() for l in curves for r in names) * 1.05); ax.set_xlim(0, TA); ax.tick_params(labelsize=6); ax.set_xlabel("t [s]", fontsize=7)
        dots.append(ax.axvline(tg[0], color="k", lw=0.8))
    cax[0].legend(fontsize=6, loc="upper right"); ttl = fig.suptitle("", fontsize=10)
    note = a.note or f"{dsp.DS} slice {Z}: same k80 input for every arm (views v%10<8, {dsp.NTV} views total), window mean +-{a.w:.0f} s, one global scale per arm (ls to the model-free late anatomy), one grey window"
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    def up(i):
        for h, (_, vv, _) in zip(hs, arms): h.set_array(vv[..., i])
        for d in dots: d.set_xdata([tg[i], tg[i]])
        ttl.set_text(f"{note}   |   t = {tg[i]:5.0f} s"); return hs + dots + [ttl]
    FuncAnimation(fig, up, frames=len(tg), interval=1000 // a.fps, blit=False).save(a.out, writer=PillowWriter(fps=a.fps)); plt.close(fig); print("saved", a.out, len(tg), "frames"); print("GIF_DONE")

if __name__ == "__main__": main()
