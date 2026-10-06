"""slide figure: the late-window model-free ruler next to the 300 s frame of every arm, HaarPSI vs that ruler under each panel, nothing else.
same pipeline as tofts_iq_track (series ls-scaled to the model-free windows, frame = window mean +-10 s, C.score on the body mask) so the
numbers equal the iq tables. --gated 1 uses the respiratory-gated late ruler. usage: DCE_DS=p3 python ruler_vs_arms_fig.py --slice 21 --items "label:path,..." --out <png>"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp, consolidated as C
from story_figs import ls_scale

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True); ap.add_argument("--gated", type=int, default=0); ap.add_argument("--t", type=float, default=300.0)
    a = ap.parse_args(); Z = a.slice; TA = dsp.TA; ctx = C.slice_ctx(Z); body = ctx["BODY"]
    ref = np.load(f"{dsp.NUF(Z)}/nufft_late{'_gated' if a.gated else ''}.npy").astype(np.float32)
    z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float)
    def refwin(nt):
        e = np.linspace(0, TA, nt + 1); o = np.zeros(mf.shape[:2] + (nt,), np.float32)
        for g in range(nt):
            m = (tmf >= e[g]) & (tmf < e[g + 1])
            if not m.any(): m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf - 0.5 * (e[g] + e[g + 1])))] = True
            o[:, :, g] = mf[:, :, m].mean(2)
        return o
    def frame(v, t, ts, w=10): m = (t > ts - w) & (t < ts + w); return v[..., m].mean(2) if m.any() else v[..., int(np.argmin(np.abs(t - ts)))]
    panels = [(f"late NUFFT ruler{' (gated)' if a.gated else ''}", ref, None)]
    for spec in [x for x in a.items.split(",") if x]:
        lab, path = spec.split(":", 1)
        if not os.path.exists(path): print("missing", path); continue
        v = np.abs(np.load(path)).astype(np.float32); t = (np.arange(v.shape[-1]) + 0.5) * TA / v.shape[-1]; v = ls_scale(v, refwin(v.shape[-1]), body); p = frame(v, t, a.t)
        panels.append((lab, p, C.score(p, ref, body)["haarpsi"]))
    n = len(panels); fig, ax = plt.subplots(1, n, figsize=(2.6 * n, 3.1)); vm = np.percentile(ref[body], 99.5)
    for k, (lab, im, h) in enumerate(panels):
        s = np.sum(im[body] * ref[body]) / (np.sum(im[body] ** 2) + 1e-12); ax[k].imshow(im * s, cmap="gray", vmin=0, vmax=vm); ax[k].axis("off")
        ax[k].set_title(lab if h is None else f"{lab}\nHaarPSI {h:.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(a.out, dpi=150, facecolor="white"); print("saved", a.out, {lab: (None if h is None else round(h, 3)) for lab, _, h in panels}); print("RULER_ARMS_DONE")

if __name__ == "__main__": main()
