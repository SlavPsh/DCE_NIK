"""sharpness vs distance from the image centre: is a recon blurred away from the centre? for every item the 90 s frame (window +-10 s, ls-scaled to
the reference) is high-passed (minus gaussian sigma 3) and the rms inside radial annuli of the body mask is divided by the same rms of the reference
(cs100 = grasp-pro all spokes) and of grasp; the intensity ratio per annulus separates blur from darkening (support prior). also a local sharpness
map (highpass rms in 15 px windows) ratio item / reference. usage: DCE_DS=p14 python radial_blur_diag.py --slice 24 --items "label:path,..." --tag x"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, argparse, json, numpy as np, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp, consolidated as C

def frame(v, t, ts, w=10):
    m = (t > ts - w) & (t < ts + w); return v[..., m].mean(2) if m.any() else v[..., int(np.argmin(np.abs(t - ts)))]

def hp(x, s=3.0): return x - ndi.gaussian_filter(x, s)

def local_rms(x, w=15): return np.sqrt(ndi.uniform_filter(x ** 2, w))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); ap.add_argument("--items", required=True); ap.add_argument("--tag", default="")
    ap.add_argument("--t", type=float, default=90.0); ap.add_argument("--step", type=int, default=16); a = ap.parse_args(); Z = a.slice; TA = dsp.TA
    ctx = C.slice_ctx(Z); body = ctx["BODY"]; cs = ctx["cs100"]; tcs = np.linspace(0, TA, cs.shape[-1]); ref = frame(cs, tcs, a.t)
    n = body.shape[0]; yy, xx = np.mgrid[:n, :n]; rad = np.sqrt((yy - n / 2 + 0.5) ** 2 + (xx - n / 2 + 0.5) ** 2)     # px from the crop centre
    edges = np.arange(0, n // 2 + a.step, a.step); bins = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    items = [("cs100 (ref)", ref)]
    for spec in [x for x in a.items.split(",") if x]:
        lab, path = spec.split(":", 1)
        if not os.path.exists(path): print("missing", path); continue
        v = np.abs(np.load(path)).astype(np.float32)
        if v.shape[:2] != body.shape: print("wrong grid, skipped", lab, v.shape); continue
        t = (np.arange(v.shape[-1]) + 0.5) * TA / v.shape[-1]; f = frame(v, t, a.t); items.append((lab, f * (np.sum(f[body] * ref[body]) / (np.sum(f[body] ** 2) + 1e-12))))   # scalar ls scale of the frame to the reference frame
    rh = hp(ref); rows = []; maps = {}
    for lab, im in items:
        h = hp(im); r = []
        for lo, hi in bins:
            m = body & (rad >= lo) & (rad < hi)
            if m.sum() < 50: r.append((np.nan, np.nan)); continue
            r.append((float(np.sqrt((h[m] ** 2).mean()) / (np.sqrt((rh[m] ** 2).mean()) + 1e-12)), float(im[m].mean() / (ref[m].mean() + 1e-12))))
        rows.append((lab, r)); maps[lab] = local_rms(h) / (local_rms(rh) + 1e-12)
    L = [f"# sharpness vs radius, {dsp.DS} slice {Z}, {a.t:.0f} s frame; fine-scale rms (highpass sigma 3) / same of cs100 per annulus of the body mask; intensity ratio in brackets", "",
         "| recon | " + " | ".join(f"{lo}-{hi} px" for lo, hi in bins) + " |", "|---|" + "---|" * len(bins)]
    for lab, r in rows: L.append(f"| {lab} | " + " | ".join("-" if np.isnan(s) else f"{s:.2f} ({q:.2f})" for s, q in r) + " |")
    g = next((r for lab, r in rows if lab.startswith("GRASP") and not lab.startswith("GRASP-Pro")), None)
    if g is not None:
        L += ["", "relative to GRASP (fine-scale ratio item / grasp per annulus):", ""]
        for lab, r in rows: L.append(f"| {lab} | " + " | ".join("-" if np.isnan(s) or np.isnan(gs) else f"{s / gs:.2f}" for (s, _), (gs, _) in zip(r, g)) + " |")
    out = f"{B}/results/tofts_vs_patlak/radial_blur{dsp.SFX}_sl{Z}{a.tag}"; open(out + ".md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
    json.dump({lab: r for lab, r in rows}, open(out + ".json", "w"))
    k = len(items); fig, ax = plt.subplots(2, max(k, 2), figsize=(3.2 * max(k, 2), 7.2)); mid = [0.5 * (lo + hi) for lo, hi in bins]
    for j, (lab, im) in enumerate(items):
        ax[0, j].imshow(im, cmap="gray", vmin=0, vmax=np.percentile(ref[body], 99.5)); ax[0, j].set_title(lab, fontsize=9); ax[0, j].axis("off")
        ax[1, j].imshow(np.clip(maps[lab], 0, 2) * body, cmap="coolwarm", vmin=0, vmax=2); ax[1, j].axis("off"); ax[1, j].set_title("local fine-scale rms / cs100 (1 = equal)", fontsize=8)
    fig.suptitle(f"{dsp.DS} slice {Z}: local sharpness ratio vs cs100 at {a.t:.0f} s (blue = smoother than the all-spoke recon, red = more fine-scale energy)", fontsize=10); fig.tight_layout()
    fig.savefig(out + "_maps.png", dpi=120, facecolor="white")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab, r in rows[1:]: ax[0].plot(mid, [s for s, _ in r], "-o", ms=3, label=lab); ax[1].plot(mid, [q for _, q in r], "-o", ms=3, label=lab)
    ax[0].axhline(1, color="0.6", lw=0.7); ax[0].set_xlabel("distance from the image centre [px]"); ax[0].set_ylabel("fine-scale rms / cs100"); ax[0].set_title("sharpness vs radius", fontsize=10); ax[0].legend(fontsize=7)
    ax[1].axhline(1, color="0.6", lw=0.7); ax[1].set_xlabel("distance from the image centre [px]"); ax[1].set_ylabel("mean intensity / cs100"); ax[1].set_title("intensity vs radius (darkening?)", fontsize=10)
    fig.suptitle(f"{dsp.DS} slice {Z}, {a.t:.0f} s, body-masked annuli", fontsize=10); fig.tight_layout(); fig.savefig(out + ".png", dpi=130, facecolor="white"); print("RADIAL_DONE")

if __name__ == "__main__": main()
