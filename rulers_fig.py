"""geometry check of the model-free rulers of one slice: the five nufft rulers (all / pre / late / pre gated / late gated) with cs100 for
comparison, a zoom on the tissue roi, the respiratory navigator with the gating weight, and the fine-scale energy of each ruler relative to
cs100 (sharpness gain from gating). usage: DCE_DS=p14 python rulers_fig.py --slice 24"""
import warnings; warnings.filterwarnings("ignore")
import sys, json, argparse, numpy as np, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp, consolidated as C

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); a = ap.parse_args(); Z = a.slice; NUF = dsp.NUF(Z)
    ctx = C.slice_ctx(Z); body = ctx["BODY"]; rois = ctx["rois"]; cs = ctx["cs100"]; meta = json.load(open(f"{NUF}/meta.json"))
    names = ["nufft_all", "nufft_pre", "nufft_pre_gated", "nufft_late", "nufft_late_gated"]; ims = {n: np.load(f"{NUF}/{n}.npy").astype(np.float32) for n in names}
    tcs = np.linspace(0, dsp.TA, cs.shape[-1]); ims["cs100 late (300 s)"] = cs[..., (tcs > 280) & (tcs < 320)].mean(2); ims["cs100 pre"] = cs[..., tcs < meta["t_pre_s"]].mean(2)
    order = ["cs100 pre", "nufft_pre", "nufft_pre_gated", "cs100 late (300 s)", "nufft_all", "nufft_late", "nufft_late_gated"]
    nav = np.load(f"{NUF}/resp_nav.npy"); wg = np.load(f"{NUF}/resp_weight.npy"); ts = np.asarray(np.load(f"{dsp.REF}/shared.npz")["view_time"]).ravel() * dsp.TA
    def hp(x): return x - ndi.gaussian_filter(x, 3.0)
    def fine(x, ref): s = np.sum(x[body] * ref[body]) / (np.sum(x[body] ** 2) + 1e-12); return float(np.sqrt((hp(x * s)[body] ** 2).mean()) / (np.sqrt((hp(ref)[body] ** 2).mean()) + 1e-12))
    c = np.argwhere(rois[dsp.T1]).mean(0).astype(int); h = 40
    fig = plt.figure(figsize=(3.0 * len(order), 9.6)); gs = fig.add_gridspec(3, len(order), height_ratios=[1.0, 0.7, 0.6])
    for j, n in enumerate(order):
        im = ims[n]; ref = ims["cs100 late (300 s)"] if "late" in n or n == "nufft_all" else ims["cs100 pre"]; vm = np.percentile(im[body], 99.5)
        ax = fig.add_subplot(gs[0, j]); ax.imshow(im, cmap="gray", vmin=0, vmax=vm); ax.axis("off"); ax.set_title(n, fontsize=9)
        ax.text(0.02, 0.02, f"fine-scale / cs100: {fine(im, ref):.2f}", color="yellow", fontsize=8, transform=ax.transAxes)
        ax = fig.add_subplot(gs[1, j]); sl = (slice(max(c[0] - h, 0), c[0] + h), slice(max(c[1] - 2 * h, 0), c[1] + 2 * h)); ax.imshow(im[sl], cmap="gray", vmin=0, vmax=vm); ax.axis("off")
        ax.contour(rois[dsp.T1][sl].astype(float), levels=[0.5], colors="lime", linewidths=0.6)
    ax = fig.add_subplot(gs[2, :4]); o = np.argsort(ts); ax.plot(ts[o], nav[o], lw=0.5, color="0.3", label="respiratory navigator (k-centre svd mode)"); ax.plot(ts[o], wg[o] * (nav.max() - nav.min()) + nav.min(), lw=0.5, color="tab:red", alpha=0.7, label="gating weight (scaled)")
    ax.set_xlim(150, 190); ax.set_xlabel("t [s] (40 s excerpt)"); ax.legend(fontsize=8); ax.set_title(f"resp {meta['resp_freq_hz']:.2f} Hz; spokes pre {meta['n_pre_spokes']} late {meta['n_late_spokes']} (nyquist {meta['nyquist_spokes']}); gated n_eff pre {meta['n_eff_pre_gated']:.0f} late {meta['n_eff_late_gated']:.0f}", fontsize=9)
    ax = fig.add_subplot(gs[2, 4:]); ax.hist(nav, bins=60, color="0.6"); ax2 = ax.twinx(); srt = np.argsort(nav); ax2.plot(nav[srt], wg[srt], color="tab:red", lw=1); ax.set_xlabel("navigator value"); ax.set_title("navigator histogram and gating weight", fontsize=9)
    fig.suptitle(f"{dsp.DS} slice {Z}: model-free rulers (ramp-dcf nufft, saved b1) vs cs100; late window t > {meta['late_t0_s']:.0f} s; soft gating weight exp(-(d/s)^2), s = {meta['gate_q']:.0f}th percentile of |nav - mode|", fontsize=10)
    fig.tight_layout(); out = f"{B}/results/realdata_nik_vs_cs_figures/figures/rulers{dsp.SFX}_sl{Z}.png"; fig.savefig(out, dpi=120, facecolor="white"); print("saved", out)
    print("fine-scale vs cs100:", {n: round(fine(ims[n], ims["cs100 late (300 s)"] if ("late" in n or n == "nufft_all") else ims["cs100 pre"]), 3) for n in order}); print("RULERS_FIG_DONE")

if __name__ == "__main__": main()
