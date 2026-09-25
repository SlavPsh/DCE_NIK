"""proposed liver / spleen / aorta rois for a liver-slab dce scan (p14), for approval before use. built from the model-free 31-spoke series only.
liver: the largest connected region of the body with moderate late enhancement (130 to 200 s) and a slow first-pass-to-late ratio, eroded;
spleen: compact posterior region with a fast (arterial-like) first-pass-to-late ratio and high late enhancement, outside the liver and the aorta,
largest such blob, eroded; aorta: from consolidated (early-enhancement top 0.5%). per slice: overlay figure on the model-free frames at 60, 120
and 300 s, mask sizes, mean curves. masks -> dsp.ROIS(Z) (rois_proposed_<ds>_sl<Z>.npz, keys liver / spleen / aorta / static).
usage: DCE_DS=p14 python roi_propose_liver.py --slices 21,24,27"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, argparse, json, numpy as np, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp
import consolidated as C
OUTD = f"{B}/results/realdata_nik_vs_cs_figures"

def grow_seed(mf, tmf, seed, radius=16, t=60.0, erode=1):
    """aorta from a user seed (row, col): connected bright region on the smoothed frame at t within the radius, filled, eroded"""
    f = ndi.gaussian_filter(mf[..., (tmf > t - 8) & (tmf < t + 8)].mean(2), 1.0); r0, c0 = seed; yy, xx = np.ogrid[:f.shape[0], :f.shape[1]]; disc = (yy - r0) ** 2 + (xx - c0) ** 2 <= radius ** 2
    loc = f[disc]; thr = 0.5 * (np.percentile(loc, 99) + np.median(loc)); m = disc & (f > thr); lab, n = ndi.label(m)
    m = (lab == lab[r0, c0]) if lab[r0, c0] > 0 else (lab == (1 + int(np.argmax(ndi.sum(np.ones_like(lab), lab, range(1, n + 1)))))); m = ndi.binary_fill_holes(m)
    return ndi.binary_erosion(m, iterations=erode) if erode else m

def grow_organ(Ls, core, seed, radius=80, tol=0.5, erode=3):
    """organ from a seed: pixels within the radius whose smoothed late enhancement is within tol of the seed's (15x15 median), connected to the seed, opened, filled, eroded"""
    r0, c0 = seed; Lm = ndi.median_filter(Ls, size=7); ref = float(np.median(Ls[r0 - 7:r0 + 8, c0 - 7:c0 + 8])); yy, xx = np.ogrid[:Ls.shape[0], :Ls.shape[1]]; disc = (yy - r0) ** 2 + (xx - c0) ** 2 <= radius ** 2
    m = core & disc & (np.abs(Lm - ref) < tol * abs(ref)); m = ndi.binary_opening(m, iterations=2); lab, n = ndi.label(m)
    m = (lab == lab[r0, c0]) if lab[r0, c0] > 0 else (lab == (1 + int(np.argmax(ndi.sum(np.ones_like(lab), lab, range(1, n + 1)))))) if n else m
    return ndi.binary_erosion(ndi.binary_fill_holes(m), iterations=erode)

def build(mf, tmf, body, ao, erode_liver=4, erode_spleen=2, liver_lo=0.35, liver_hi=0.97, spleen_q=0.90, aorta_seed=None, liver_seed=None, spleen_seed=None):
    if aorta_seed is not None: ao = grow_seed(mf, tmf, aorta_seed)
    base = mf[..., tmf < 45].mean(2); late = mf[..., (tmf > 130) & (tmf < 200)].mean(2) - base; first = mf[..., (tmf > 60) & (tmf < 95)].mean(2) - base
    Ls = ndi.gaussian_filter(late * body, 1.5); Fs = ndi.gaussian_filter(first * body, 1.5); ratio = np.where(body, Fs / (np.abs(Ls) + 1e-9), 0.0)
    ref = np.quantile(Ls[body], 0.995)
    # aorta, redone for this resolution: early-enhancing blobs (first-pass top 1% of the body interior) with a plausible cross-section
    # (60 to 900 px at 1.5 mm / px) and roundness; among them the fastest ratio. the consolidated pick (0.5% top, largest blob) is kept as fallback.
    inner = ndi.binary_erosion(body, iterations=4); ea = inner & (Fs > np.quantile(Fs[inner], 0.99)); ea = ndi.binary_opening(ea, iterations=1); labA, nA = ndi.label(ea); best = None
    for i in range(1, nA + 1):
        m = labA == i; area = int(m.sum())
        if not 60 <= area <= 900: continue
        m = ndi.binary_fill_holes(m); rr, cc = np.nonzero(m); rad = np.sqrt(area / np.pi); round_ = area / (np.pi * max(np.ptp(rr), np.ptp(cc), 1) ** 2 / 4)
        score = float(ratio[m].mean()) * min(round_, 1.0)
        if best is None or score > best[0]: best = (score, m)
    if best is not None and aorta_seed is None: ao = best[1]
    # alternative aorta candidate for the user's decision: the roundest bright blob of the 60 s frame (top 3% intensity), area 150 to 1500 px, central third of the fov
    f60 = mf[..., (tmf > 52) & (tmf < 68)].mean(2) * inner; eb = inner & (f60 > np.quantile(f60[inner], 0.97)); eb = ndi.binary_opening(eb, iterations=2); labB, nB = ndi.label(eb); alt = None
    H, W = body.shape
    for i in range(1, nB + 1):
        m = ndi.binary_fill_holes(labB == i); area = int(m.sum()); rr, cc = np.nonzero(m)
        if not 150 <= area <= 1500 or not (W / 4 < cc.mean() < 3 * W / 4): continue
        round_ = area / (np.pi * max(np.ptp(rr), np.ptp(cc), 1) ** 2 / 4)
        if alt is None or round_ > alt[0]: alt = (round_, m)
    aorta_alt = alt[1] if alt is not None else np.zeros_like(body)
    core = ndi.binary_erosion(body, iterations=6) & ~ndi.binary_dilation(ao, iterations=6) & ~ndi.binary_dilation(aorta_alt, iterations=6)
    # liver: moderate late enhancement, slow ratio (portal-dominated), largest blob
    cand = core & (Ls > liver_lo * ref) & (Ls < liver_hi * ref) & (ratio < np.quantile(ratio[core], 0.6))
    cand = ndi.binary_opening(cand, iterations=2); lab, n = ndi.label(cand); sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1)) if n else []
    liver = (lab == (1 + int(np.argmax(sz)))) if n else cand; liver = ndi.binary_fill_holes(liver); liver = ndi.binary_erosion(liver, iterations=erode_liver)
    # spleen: fast ratio and high late enhancement, outside liver + aorta, largest compact blob
    # spleen: enhances at least like the liver, faster ratio than the liver, outside liver + aorta; vessels removed by a 3-px opening; largest blob
    rl = np.median(ratio[liver]) if liver.any() else np.quantile(ratio[core], 0.5)
    cand2 = core & ~ndi.binary_dilation(liver, iterations=6) & (Ls > 0.45 * ref) & (ratio > rl)
    cand2 = ndi.binary_opening(cand2, iterations=3); lab2, n2 = ndi.label(cand2); sz2 = ndi.sum(np.ones_like(lab2), lab2, range(1, n2 + 1)) if n2 else []
    spleen = (lab2 == (1 + int(np.argmax(sz2)))) if n2 else cand2; spleen = ndi.binary_fill_holes(spleen); spleen = ndi.binary_erosion(spleen, iterations=erode_spleen)
    if liver_seed is not None: liver = grow_organ(Ls, core, liver_seed, erode=erode_liver)
    if spleen_seed is not None: spleen = grow_organ(Ls, core & ~ndi.binary_dilation(liver, iterations=4), spleen_seed, erode=erode_spleen)
    static = ndi.binary_erosion(body, iterations=8) & (np.abs(Ls) < 0.15 * ref) & ~liver & ~spleen                                     # non-enhancing interior (muscle / fat)
    lab3, n3 = ndi.label(static); sz3 = ndi.sum(np.ones_like(lab3), lab3, range(1, n3 + 1)) if n3 else []; static = (lab3 == (1 + int(np.argmax(sz3)))) if n3 else static
    return dict(liver=liver, spleen=spleen, aorta=ao, aorta_alt=aorta_alt, static=static), dict(late=late, first=first, ratio=ratio)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slices", default="21,24,27"); ap.add_argument("--erode-liver", type=int, default=4); ap.add_argument("--erode-spleen", type=int, default=2); ap.add_argument("--aorta-seed", default="", help="per-slice row,col seeds: 21:145,137;24:..."); ap.add_argument("--liver-seed", default=""); ap.add_argument("--spleen-seed", default="")
    a = ap.parse_args(); P_ = lambda t: {int(k): tuple(int(v) for v in rc.split(",")) for k, rc in (x.split(":") for x in t.split(";") if x)}; seeds, lseeds, sseeds = P_(a.aorta_seed), P_(a.liver_seed), P_(a.spleen_seed)
    rep = {}; L = ["# proposed liver / spleen rois (for approval; nothing uses them yet)", "", "| slice | liver px | spleen px | aorta px | static px | liver ttp s / plateau | spleen ttp s / plateau |", "|---|---|---|---|---|---|---|"]
    for Z in [int(s) for s in a.slices.split(",")]:
        ctx = C.slice_ctx(Z); body = ctx["BODY"]; ao = ctx["rois"]["aorta"]; z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float)
        rois, aux = build(mf, tmf, body, ao, a.erode_liver, a.erode_spleen, aorta_seed=seeds.get(Z), liver_seed=lseeds.get(Z), spleen_seed=sseeds.get(Z))
        cur = {r: np.array([mf[..., i][rois[r]].mean() for i in range(mf.shape[-1])]) if rois[r].any() else np.zeros(mf.shape[-1]) for r in rois}; cur = {r: c - np.median(c[tmf < 40]) for r, c in cur.items()}
        ttp = {r: float(tmf[np.argmax(cur[r])]) for r in cur}; plat = {r: float(cur[r][tmf > 250].mean() / (cur[r].max() + 1e-9)) for r in cur}
        rep[Z] = dict(px={r: int(rois[r].sum()) for r in rois}, ttp=ttp, plateau=plat); np.savez(dsp.ROIS(Z), **rois, source="model-free 31-spoke nufft, liver / spleen proposal v4", aorta_tag=np.array(f"seed {seeds.get(Z)}" if Z in seeds else "auto"))
        L.append(f"| {Z} | {rois['liver'].sum()} | {rois['spleen'].sum()} | {rois['aorta'].sum()} | {rois['static'].sum()} | {ttp['liver']:.0f} / {plat['liver']:.2f} | {ttp['spleen']:.0f} / {plat['spleen']:.2f} |")
        frame = lambda ts, w=8: mf[..., (tmf > ts - w) & (tmf < ts + w)].mean(2)
        fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
        for k, (ttl, im) in enumerate((("late enhancement 130 to 200 s", aux["late"]), ("first-pass / late ratio", aux["ratio"]), ("model-free at 60 s", frame(60)), ("model-free at 120 s", frame(120)))):
            ax[k].imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im[body], 99.5) if im[body].max() > 0 else 1); ax[k].set_title(ttl, fontsize=9)
            ax[k].set_xticks(range(0, im.shape[1], 20)); ax[k].set_yticks(range(0, im.shape[0], 20)); ax[k].grid(color="yellow", alpha=0.2, lw=0.4); ax[k].tick_params(labelsize=5)
            for r, c in (("liver", "lime"), ("spleen", "orange"), ("aorta", "cyan"), ("static", "magenta")):
                if rois[r].any(): ax[k].contour(rois[r].astype(float), levels=[0.5], colors=[c], linewidths=0.9)
        for r, c in (("aorta", "cyan"), ("liver", "lime"), ("spleen", "orange"), ("static", "magenta")): ax[4].plot(tmf, cur[r] / (cur["aorta"].max() + 1e-9), color=c, lw=1.4, label=f"{r} ({int(rois[r].sum())} px)")
        ax[4].set_title("model-free mean curves, proposed rois (norm to aorta peak)", fontsize=9); ax[4].set_xlabel("t [s]"); ax[4].legend(fontsize=7); ax[4].axhline(0, color="0.7", lw=0.6)
        fig.suptitle(f"{dsp.DS} slice {Z}: proposed liver (lime) / spleen (orange) / aorta (cyan) / static (magenta); grid every 20 px (row, col)", fontsize=11); fig.tight_layout()
        fig.savefig(f"{OUTD}/figures/roi_proposed{dsp.SFX}_sl{Z}.png", dpi=130, facecolor="white"); plt.close(fig); print("saved", Z, rep[Z], flush=True)
    json.dump(rep, open(f"{OUTD}/rois_proposed{dsp.SFX}.json", "w"), indent=1); open(f"{OUTD}/rois_proposed{dsp.SFX}.md", "w").write("\n".join(L)); print("\n".join(L)); print("ROI_PROPOSE_DONE")

if __name__ == "__main__": main()
