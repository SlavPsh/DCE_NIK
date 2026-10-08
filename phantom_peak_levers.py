"""phantom (truth) check of the first-pass peak levers: per-atom weight decay on the tofts amplitude head, atom scale, wd level.
reads the xph_eval npz of each arm (rec_best = global truth-scaled, rot180-aligned), reports per roi the peak ratio vs truth under the
global scale and after matching the late plateau (90 to 200 s, isolates the first-pass deficit from the overall amplitude), curve nrmse,
image metrics vs truth, test k-space nmse. usage: --items "label:tag,..." --out <md> --fig <png>"""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import xph_pipeline as P, xph_common as X
from masked_metrics import haarpsi_masked, ssim_masked
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--items", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fig", required=True)
ap.add_argument("--title", default=""); a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"])
MT = torch.from_numpy(body.astype(np.float32))[None, None].to(dev); LATE = (tq >= 90) & (tq < 200)
ROIS = ("aorta", "cortex", "medulla")

def peak(c): return float((c - np.median(c[:8])).max())                       # baseline-subtracted first-pass peak
def img_metrics(img, ref):
    hp, ss = [], []
    for t in range(0, img.shape[-1], max(1, img.shape[-1] // 40)):
        vmax = float(np.percentile(ref[:, :, t][body], 99.5)) + 1e-12
        x = torch.from_numpy(np.clip(img[:, :, t] / vmax, 0, 1)[None, None]).float().to(dev); y = torch.from_numpy(np.clip(ref[:, :, t] / vmax, 0, 1)[None, None]).float().to(dev)
        hp.append(float(haarpsi_masked(x, y, MT, data_range=1.0).cpu())); ss.append(float(ssim_masked(x, y, MT, data_range=1.0).cpu()))
    return float(np.mean(hp)), float(np.mean(ss))

rows, curves = [], {}
for it in a.items.split(","):
    lab, tag = it.split(":", 1); f = f"{P.OUT}/arrays/nik_eval_{tag}.npz"
    if not os.path.exists(f): print("missing", f); continue
    e = np.load(f, allow_pickle=True); rec = e["rec_best"].astype(np.float32); r = dict(label=lab, tag=tag, best_step=int(e["best_step"]), scale=float(e["scale_best"]),
        test_knmse=float(np.asarray(e["test_best"])[0]), img_nrmse=float(e["img_nrmse_mean_best"]))
    r["haarpsi"], r["ssim"] = img_metrics(rec, Tr); curves[lab] = {}
    for roi in ROIS:
        c = e[f"{roi}_curve_rec"]; ct = e[f"{roi}_curve_true"]; curves[lab][roi] = (c, ct)
        r[f"{roi}_nrmse"] = float(np.linalg.norm(c - ct) / np.linalg.norm(ct)); r[f"{roi}_peak"] = peak(c) / peak(ct)
        g = float((ct[LATE] - np.median(ct[:8])).mean() / ((c[LATE] - np.median(c[:8])).mean() + 1e-12)); r[f"{roi}_peak_pm"] = peak(c * g) / peak(ct)   # plateau-matched
    rows.append(r); print(lab, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)

cols = ["best_step", "scale", "cortex_peak", "cortex_peak_pm", "medulla_peak", "medulla_peak_pm", "aorta_peak", "aorta_peak_pm", "cortex_nrmse", "medulla_nrmse", "aorta_nrmse", "haarpsi", "ssim", "img_nrmse", "test_knmse"]
L = [f"# {a.title}", "", "peak = baseline-subtracted first-pass peak / truth peak under the ONE global truth-derived image scale (xph_eval); peak_pm = same after matching the",
     "90 to 200 s plateau to truth (first-pass deficit only). scale = global factor the recon needed (1 = correct amplitude). image metrics vs truth over the body, 40 frames.", "",
     "| arm | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
for r in rows: L.append(f"| {r['label']} | " + " | ".join(f"{r[c]:.3g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
os.makedirs(os.path.dirname(a.out), exist_ok=True); open(a.out, "w").write("\n".join(L) + "\n"); print("wrote", a.out)

fig, ax = plt.subplots(2, 3, figsize=(16, 8)); cm = plt.get_cmap("tab10")
for j, roi in enumerate(ROIS):
    for i, (lab, cv) in enumerate(curves.items()):
        c, ct = cv[roi]
        for k, (lo, hi) in enumerate(((0, tq[-1]), (10, 70))):
            m = (tq >= lo) & (tq <= hi); ax[k, j].plot(tq[m], c[m], color=cm(i % 10), lw=1.2, label=lab if (j == 0 and k == 0) else None)
    for k in range(2):
        lo, hi = ((0, tq[-1]), (10, 70))[k]; m = (tq >= lo) & (tq <= hi); ax[k, j].plot(tq[m], cv[roi][1][m], "k--", lw=1.5, label="truth" if (j == 0 and k == 0) else None)
        ax[k, j].set_title(f"{roi}" + (" (first pass)" if k else "")); ax[k, j].set_xlabel("s"); ax[k, j].grid(alpha=0.3)
ax[0, 0].legend(fontsize=8); fig.suptitle(a.title, fontsize=10); fig.tight_layout(); fig.savefig(a.fig, dpi=120); print("wrote", a.fig)
