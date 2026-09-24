"""decision figure for the aorta on a new dataset: the model-free frame at t_show with every early-enhancing blob numbered, their areas and their
enhancement curves, so the user can name the aorta by number. usage: DCE_DS=p14 python roi_candidates_fig.py --slice 21 [--t 60 --q 0.97 --min-px 30]"""
import warnings; warnings.filterwarnings("ignore")
import sys, argparse, numpy as np, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp, consolidated as C

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, default=21); ap.add_argument("--t", type=float, default=60.0); ap.add_argument("--q", type=float, default=0.97); ap.add_argument("--min-px", type=int, default=30); ap.add_argument("--max", type=int, default=12); a = ap.parse_args(); Z = a.slice
    ctx = C.slice_ctx(Z); body = ctx["BODY"]; z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float)
    base = mf[..., tmf < 45].mean(2); f = mf[..., (tmf > a.t - 8) & (tmf < a.t + 8)].mean(2); early = (f - base) * body
    inner = ndi.binary_erosion(body, iterations=3); m = inner & (early > np.quantile(early[inner], a.q)); m = ndi.binary_opening(m, iterations=1); lab, n = ndi.label(m)
    blobs = sorted([(int((lab == i).sum()), i) for i in range(1, n + 1) if (lab == i).sum() >= a.min_px], reverse=True)[:a.max]
    # bright-at-baseline vessels (inflow): round blobs of the raw frame's top 2%, area 100 to 1500 px, appended as extra candidates
    yy, xx = np.ogrid[-22:23, -22:23]; disk = (xx ** 2 + yy ** 2) <= 22 ** 2                                             # top-hat: round bright structures smaller than the disk (vessels), independent of the absolute level
    th = ndi.white_tophat(ndi.gaussian_filter(f, 1.0) * inner, footprint=disk); mb = inner & (th > np.quantile(th[inner], 0.985)); mb = ndi.binary_opening(mb, iterations=2); labB, nB = ndi.label(mb); nxt = lab.max() + 1
    for i in range(1, nB + 1):
        m2 = ndi.binary_fill_holes(labB == i); area = int(m2.sum()); rr, cc = np.nonzero(m2)
        if not 100 <= area <= 1500: continue
        if (lab[m2] > 0).mean() > 0.5: continue                                                       # already an early-enhancing blob
        lab[m2 & (lab == 0)] = nxt; blobs.append((area, nxt)); nxt += 1
    blobs = blobs[:a.max + 4]
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2)); vm = np.percentile(f[body], 99.5)
    ax[0].imshow(f, cmap="gray", vmin=0, vmax=vm); ax[0].set_title(f"model-free at {a.t:.0f} s, early-enhancing blobs (top {100 * (1 - a.q):.0f}%) numbered", fontsize=10); ax[0].axis("off")
    ax[1].imshow(early, cmap="gray", vmin=0, vmax=np.percentile(early[body], 99.5)); ax[1].set_title("enhancement at that time (frame minus baseline)", fontsize=10); ax[1].axis("off")
    cols = plt.cm.tab10(np.linspace(0, 1, 10)); curves = []
    for k, (area, i) in enumerate(blobs):
        mk = ndi.binary_fill_holes(lab == i); rr, cc = np.nonzero(mk); c = np.array([mf[..., j][mk].mean() for j in range(mf.shape[-1])]); c = c - np.median(c[tmf < 40]); curves.append((k + 1, area, c))
        for axx in ax[:2]: axx.contour(mk.astype(float), levels=[0.5], colors=[cols[k % 10]], linewidths=0.9); axx.text(cc.mean(), rr.mean(), str(k + 1), color="yellow", fontsize=11, fontweight="bold", ha="center", va="center")
    pk = max(c.max() for _, _, c in curves) if curves else 1.0
    for k, area, c in curves: ax[2].plot(tmf, c / pk, color=cols[(k - 1) % 10], lw=1.3, label=f"{k}: {area} px, ttp {tmf[np.argmax(c)]:.0f} s")
    ax[2].set_title("candidate curves (norm to the highest peak)", fontsize=10); ax[2].set_xlabel("t [s]"); ax[2].legend(fontsize=8); ax[2].axhline(0, color="0.7", lw=0.6)
    fig.suptitle(f"{dsp.DS} slice {Z}: which numbered blob is the aorta?", fontsize=12); fig.tight_layout(); out = f"{B}/results/realdata_nik_vs_cs_figures/figures/aorta_candidates{dsp.SFX}_sl{Z}.png"; fig.savefig(out, dpi=130, facecolor="white"); print("saved", out)
    np.savez(f"{B}/results/realdata_nik_vs_cs_figures/aorta_candidates{dsp.SFX}_sl{Z}.npz", lab=lab, ids=np.array([i for _, i in blobs]), areas=np.array([ar for ar, _ in blobs])); print("CANDIDATES_DONE")

if __name__ == "__main__": main()
