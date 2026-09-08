"""Per-frame HaarPSI vs CS-100 (full-spoke) reference, spoke-reduction sweep.
NIK (342 fr) resampled to the CS frame grid (122). Reports mean + min (worst frame).
Q: as spokes drop, does NIK's similarity-to-full-spoke-CS hold while CS's falls?
out: haarpsi_spoke.json + haarpsi_spoke.png"""
import numpy as np, json, os, torch, piq
from scipy.interpolate import interp1d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figpath import fig as fpath
D = "/scratch/rnga/vvpshenov/DCE_NIK"
# reference-method plumbing. defaults = grasp-pro (unchanged). set CSD/CSPRE/TAG to swap in grasp v2:
#   CSD=/scratch/rnga/vvpshenov/grasp_v2/results_grasp_v2 CSPRE=gv2 TAG=_gv2
CSD = os.environ.get("CSD", "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs")
CSPRE = os.environ.get("CSPRE", "cs"); TAG = os.environ.get("TAG", "")
LABS = [("f100", 100, 14), ("f70", 71, 10), ("f50", 50, 7), ("f35", 36, 5), ("f25", 29, 4)]
dev = "cuda" if torch.cuda.is_available() else "cpu"

ref = np.abs(np.load(f"{CSD}/{CSPRE}_slice13_f100.npy")).astype(np.float32)   # [x,y,122] anchor
NT = ref.shape[-1]
rm = ref.mean(-1); body = rm > np.quantile(rm, 0.55)                      # body ROI (zero outside)

def resample_t(v, nt):
    if v.shape[-1] == nt: return v
    src = np.linspace(0, 1, v.shape[-1]); dst = np.linspace(0, 1, nt)
    return interp1d(src, v, axis=-1, kind="linear")(dst).astype(np.float32)

def to_batch(v):
    """[x,y,T] -> [T,1,x,y] in [0,1], ROI-masked, per-volume scaling (metric is scale-sensitive)."""
    v = v * body[:, :, None]
    v = v / (np.percentile(v, 99.5) + 1e-9)
    v = np.clip(v, 0, 1)
    return torch.from_numpy(np.transpose(v, (2, 0, 1))[:, None]).float().to(dev)

rb = to_batch(ref)
res = {}
print(f"{'run':14} {'HaarPSI mean':>12} {'min':>7}")
for lab, pct, spf in LABS:
    for meth, path in [("NIK", f"{D}/results_spoke_nik_{lab}/nik_slice_13.npy"),
                       ("CS",  f"{CSD}/{CSPRE}_slice13_{lab}.npy")]:
        if not os.path.exists(path): continue
        v = resample_t(np.abs(np.load(path)).astype(np.float32), NT)
        with torch.no_grad():
            h = piq.haarpsi(to_batch(v), rb, reduction="none", data_range=1.0).cpu().numpy()
        res[f"{meth}_{lab}"] = dict(meth=meth, pct=pct, spf=spf, mean=float(h.mean()), min=float(h.min()))
        print(f"{meth+' '+lab:14} {h.mean():12.4f} {h.min():7.4f}", flush=True)

json.dump(res, open(f"{D}/haarpsi_spoke{TAG}.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(6.6, 4.3))
for meth, col in [("NIK", "#7c3aed"), ("CS", "#0369a1")]:
    r = sorted([v for v in res.values() if v["meth"] == meth], key=lambda z: -z["pct"])
    x = [z["pct"] for z in r]
    ax.plot(x, [z["mean"] for z in r], "o-", color=col, lw=2.2, ms=7, label=f"{meth} mean")
    ax.plot(x, [z["min"] for z in r], "^--", color=col, lw=1.2, ms=5, alpha=.55, label=f"{meth} worst frame")
ax.invert_xaxis(); ax.set_xlabel("spokes retained (%)"); ax.set_ylabel("per-frame HaarPSI vs CS-100")
ax.set_title("Spoke reduction: who holds up?", fontsize=11); ax.grid(alpha=.3); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(fpath(f"haarpsi_spoke{TAG}.png"), dpi=150, bbox_inches="tight")
print(f"wrote haarpsi_spoke{TAG}.json/.png")
