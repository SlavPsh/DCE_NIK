"""proposed anatomical kidney rois for the in vivo k80 comparison, for approval before use (nothing reads them yet).
built from the model-free 31-spoke nufft series only (no method under test): kidney = late enhancement (130 to 200 s) above a body-relative
threshold, two largest blobs, holes filled, one-pixel erosion against edge partial volume; cortex / medulla split inside the kidney by the
first-pass to late enhancement ratio (cortex enhances first, medulla later); aorta kept from consolidated. per slice: overlay figure on the
model-free frames at 90 s and 300 s and on grasp, mask sizes, mean model-free curves per roi. masks saved to rois_proposed_sl<Z>.npz.
usage: python roi_propose_invivo.py --slices 21,18,19 [--thr 0.35 --erode 1]"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, argparse, json, numpy as np, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; GV = "/net/beegfs/users/P101440/grasp_v2/results_grasp_v2"; sys.path.insert(0, B)
import consolidated as C
OUTD = f"{B}/results/realdata_nik_vs_cs_figures"

def build(mf, tmf, body, thr, erode, min_px=120):
    base = mf[..., tmf < 45].mean(2); late = mf[..., (tmf > 130) & (tmf < 200)].mean(2) - base; first = mf[..., (tmf > 65) & (tmf < 100)].mean(2) - base
    tail = mf[..., tmf > 250].mean(2) - base
    ref = np.quantile(late[body], 0.995); kid = body & (late > thr * ref); kid = ndi.binary_opening(kid, iterations=1); kid = ndi.binary_fill_holes(kid)
    lab, n = ndi.label(kid); sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1)) if n else []
    keep = [i + 1 for i in np.argsort(sizes)[::-1][:2] if sizes[i] >= min_px]; kid = np.isin(lab, keep)
    if erode: kid = ndi.binary_erosion(kid, iterations=erode)
    ratio = np.where(kid, first / (tail + 1e-9), 0.0); thr_r = np.median(ratio[kid]) if kid.any() else 0
    cortex = kid & (ratio > thr_r); medulla = kid & (ratio <= thr_r)
    return dict(kidney=kid, cortex=cortex, medulla=medulla), dict(late=late, first=first, tail=tail, ratio=ratio, n_blobs=len(keep))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slices", default="21,18,19"); ap.add_argument("--thr", type=float, default=0.35); ap.add_argument("--erode", type=int, default=1); a = ap.parse_args(); TA = 375.0
    rep = {}
    for Z in [int(s) for s in a.slices.split(",")]:
        ctx = C.slice_ctx(Z); body = ctx["BODY"]; old = ctx["rois"]; z = np.load(f"{B}/step2_slice{Z}.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float)
        rois, aux = build(mf, tmf, body, a.thr, a.erode); rois["aorta"] = old["aorta"]
        gv = f"{GV}/gv2_slice{Z}_n12_k80.npy"; g = np.abs(np.load(gv)).astype(np.float32) if os.path.exists(gv) else None
        frame = lambda v, t, ts, w=8: v[..., (t > ts - w) & (t < ts + w)].mean(2)
        curves = {r: np.array([mf[..., i][rois[r]].mean() for i in range(mf.shape[-1])]) for r in ("aorta", "cortex", "medulla", "kidney")}
        curves = {r: c - np.median(c[tmf < 40]) for r, c in curves.items()}
        rep[Z] = dict(px={r: int(rois[r].sum()) for r in rois}, blobs=aux["n_blobs"], old_px={r: int(old[r].sum()) for r in old if r in ("cortex", "medulla", "aorta")},
                      ttp_s={r: float(tmf[np.argmax(curves[r])]) for r in curves}, plateau_over_peak={r: float(curves[r][tmf > 250].mean() / (curves[r].max() + 1e-9)) for r in curves})
        np.savez(f"{OUTD}/rois_proposed_sl{Z}.npz", **{r: rois[r] for r in rois}, thr=a.thr, erode=a.erode, source="model-free 31-spoke nufft, late enhancement 130-200 s")
        fig, ax = plt.subplots(1, 5, figsize=(21, 4.6)); vm = np.percentile(mf[..., 0][body], 99.5)
        panels = [("late enhancement 130-200 s (model-free), kidney = 2 largest blobs", aux["late"], None), ("model-free at 90 s: cortex (green) / medulla (orange)", frame(mf, tmf, 90), None),
                  ("model-free at 300 s", frame(mf, tmf, 300), None), ("GRASP k80 at 90 s" if g is not None else "", frame(g, (np.arange(g.shape[-1]) + 0.5) * TA / g.shape[-1], 90) if g is not None else np.zeros_like(body, float), None)]
        for k, (ttl, im, _) in enumerate(panels):
            ax[k].imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im[body], 99.5) if im.any() else 1); ax[k].set_title(ttl, fontsize=9); ax[k].axis("off")
            for r, c in (("kidney", "yellow"), ("cortex", "lime"), ("medulla", "orange"), ("aorta", "cyan")):
                if rois[r].any() and not (k == 0 and r != "kidney"): ax[k].contour(rois[r].astype(float), levels=[0.5], colors=[c], linewidths=0.9)
            if k == 1:
                for r in ("cortex", "medulla"):
                    if old[r].any(): ax[k].contour(old[r].astype(float), levels=[0.5], colors=["red"], linewidths=0.6, linestyles="dotted")
                ax[k].plot([], [], color="red", ls=":", label="old crescent rois"); ax[k].legend(fontsize=7, loc="lower left")
        for r, c in (("aorta", "cyan"), ("cortex", "lime"), ("medulla", "orange"), ("kidney", "yellow")): ax[4].plot(tmf, curves[r] / (curves["aorta"].max() + 1e-9), color=c, lw=1.4, label=f"{r} ({int(rois[r].sum())} px)")
        ax[4].set_title("model-free mean curves, proposed rois (norm to aorta peak)", fontsize=9); ax[4].set_xlabel("t [s]"); ax[4].legend(fontsize=7); ax[4].axhline(0, color="0.7", lw=0.6)
        fig.suptitle(f"slice {Z}: proposed kidney rois (thr {a.thr} of the 99.5th percentile late enhancement, erode {a.erode} px) vs the old crescent rois", fontsize=11); fig.tight_layout()
        fig.savefig(f"{OUTD}/figures/roi_proposed_sl{Z}.png", dpi=130, facecolor="white"); plt.close(fig); print("saved", Z, rep[Z], flush=True)
    json.dump(rep, open(f"{OUTD}/rois_proposed.json", "w"), indent=1)
    L = ["# proposed kidney rois (for approval; nothing uses them yet)", "", "| slice | kidney px | cortex px | medulla px | blobs | old cortex / medulla px | ttp cortex / medulla (s) | plateau/peak cortex / medulla |", "|---|---|---|---|---|---|---|---|"]
    for Z, r in rep.items(): L.append(f"| {Z} | {r['px']['kidney']} | {r['px']['cortex']} | {r['px']['medulla']} | {r['blobs']} | {r['old_px']['cortex']} / {r['old_px']['medulla']} | {r['ttp_s']['cortex']:.0f} / {r['ttp_s']['medulla']:.0f} | {r['plateau_over_peak']['cortex']:.2f} / {r['plateau_over_peak']['medulla']:.2f} |")
    open(f"{OUTD}/rois_proposed.md", "w").write("\n".join(L)); print("\n".join(L)); print("ROI_PROPOSE_DONE")

if __name__ == "__main__": main()
