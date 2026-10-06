"""presentation figure of the model-free rulers of one slice: pre-contrast nufft (ungated / gated), late-window nufft (ungated / gated) and the
cs100 late frame, full image and a zoom on the tissue roi, with spoke counts and the fine-scale energy relative to cs100.
usage: DCE_DS=p14 python rulers_slide.py --slice 24"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, argparse, numpy as np, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp, consolidated as C

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); a = ap.parse_args(); Z = a.slice; NUF = dsp.NUF(Z)
    ctx = C.slice_ctx(Z); body = ctx["BODY"]; rois = ctx["rois"]; cs = ctx["cs100"]; meta = json.load(open(f"{NUF}/meta.json")); tcs = np.linspace(0, dsp.TA, cs.shape[-1])
    cs_late = cs[..., (tcs > 280) & (tcs < 320)].mean(2)
    items = [("pre-contrast NUFFT\n(%d spokes, t < %.0f s)" % (meta["n_pre_spokes"], meta["t_pre_s"]), np.load(f"{NUF}/nufft_pre.npy")), ("pre-contrast NUFFT, gated\n(n_eff %.0f spokes)" % meta["n_eff_pre_gated"], np.load(f"{NUF}/nufft_pre_gated.npy")),
             ("late NUFFT, t > %.0f s\n(%d spokes, nyquist %d)" % (meta["late_t0_s"], meta["n_late_spokes"], meta["nyquist_spokes"]), np.load(f"{NUF}/nufft_late.npy")), ("late NUFFT, gated\n(n_eff %.0f spokes)" % meta["n_eff_late_gated"], np.load(f"{NUF}/nufft_late_gated.npy")),
             ("GRASP-Pro all spokes (cs100)\n300 s frame, the old ruler", cs_late)]
    def hp(x): return x - ndi.gaussian_filter(x, 3.0)
    def fine(x, ref): s = np.sum(x[body] * ref[body]) / (np.sum(x[body] ** 2) + 1e-12); return float(np.sqrt((hp(x * s)[body] ** 2).mean()) / (np.sqrt((hp(ref)[body] ** 2).mean()) + 1e-12))
    c = np.argwhere(rois[dsp.T1]).mean(0).astype(int); h = 40; sl = (slice(max(c[0] - h, 0), c[0] + h), slice(max(c[1] - 2 * h, 0), c[1] + 2 * h))
    fig, ax = plt.subplots(2, len(items), figsize=(3.3 * len(items), 6.6), gridspec_kw=dict(height_ratios=[1.0, 0.55]))
    for j, (lab, im) in enumerate(items):
        im = im.astype(np.float32); vm = np.percentile(im[body], 99.5); ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=vm); ax[0, j].axis("off"); ax[0, j].set_title(lab, fontsize=9)
        ax[0, j].text(0.02, 0.02, f"fine-scale vs cs100 {fine(im, cs_late):.2f}", color="yellow", fontsize=8, transform=ax[0, j].transAxes)
        ax[1, j].imshow(im[sl], cmap="gray", vmin=0, vmax=vm); ax[1, j].axis("off"); ax[1, j].contour(rois[dsp.T1][sl].astype(float), levels=[0.5], colors="lime", linewidths=0.6)
    fig.suptitle(f"{dsp.DS} slice {Z}: model-free image-quality rulers (ramp-dcf NUFFT, saved coil maps, no regularization, no subspace); respiratory gating from the k-centre navigator ({meta['resp_freq_hz']:.2f} Hz)", fontsize=10)
    fig.tight_layout(); out = f"{B}/results/realdata_nik_vs_cs_figures/figures/rulers_slide{dsp.SFX}_sl{Z}.png"; fig.savefig(out, dpi=130, facecolor="white"); print("saved", out); print("RULERS_SLIDE_DONE")

if __name__ == "__main__": main()
