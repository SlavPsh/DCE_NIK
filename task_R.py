"""TASK R: redo the peak-normalization-contaminated temporal numbers on RAW curves with a
SINGLE global affine per method (fit across ALL ROIs simultaneously; per-ROI fitting is what
hid the global offset and faked a cortex anomaly). reference = streak-free model-free ROI-mean.
methods: CS(f25), NIK R16, NIK full-rank. ROIs: aorta/cortex/medulla/liver. slices 18,19,21.
out: task_R.json + figure + printed tables."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, sys
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); sys.path.insert(0, ".")
from figpath import fig as fpath
import consolidated as C
D = "/scratch/rnga/vvpshenov/DCE_NIK"; REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
CSD = "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs"; TA = 375.0
SLICES = [18, 19, 21]; ROIS = ["aorta", "cortex", "medulla", "liver"]
def nikpath(kind, Z):
    if kind == "NIK_R16": return f"{D}/results_batch/nik_r16_sl{Z}/nik_slice_{Z}_cplx.npy"
    if kind == "NIK_full": return f"{D}/results_batch/full_sl{Z}{'f25' if Z==21 else ''}/nik_slice_{Z}_cplx.npy"
def relres(p, t): return float(np.sqrt(np.mean((p - t) ** 2)) / (t.max() - t.min() + 1e-9))
def osc(c):
    sm = savgol_filter(c, 11, 3); return float(np.std(c - sm) / (abs(sm).max() + 1e-9))
def plateau(t, c): m = (t > 100) & (t < 210); return float(np.median(c[m]))
def firstpass_peak(t, c): m = (t > 20) & (t < 110); return float(savgol_filter(c, 11, 3)[m].max())

out = {}
for Z in SLICES:
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; mf = np.load(f"{D}/step2_slice{Z}.npz")["mf"]; tmf = ctx["tmf"]
    names = [r for r in ROIS if rois.get(r) is not None and rois[r].sum() > 0]
    real = {r: np.array([im[rois[r]].mean() for im in mf]) for r in names}         # RAW streak-free reference, on tmf
    methods = {}
    for kind in ["CS", "NIK_R16", "NIK_full"]:
        if kind == "CS":
            v = np.abs(np.load(f"{CSD}/cs_slice{Z:02d}_f25.npy")).astype(np.float32); tv = np.linspace(0, TA, v.shape[-1])
        else:
            v = np.abs(np.load(nikpath(kind, Z))).astype(np.float32); tv = np.linspace(0, TA, v.shape[-1])
        cur = {r: np.interp(tmf, tv, np.array([v[..., i][rois[r]].mean() for i in range(v.shape[-1])])) for r in names}  # RAW, resampled to tmf
        # ONE global affine (a,b) across ALL ROIs simultaneously: minimize ||a*method + b - real||
        Ns = np.concatenate([cur[r] for r in names]); Rs = np.concatenate([real[r] for r in names])
        A = np.stack([Ns, np.ones_like(Ns)], 1); (a, b), *_ = np.linalg.lstsq(A, Rs, rcond=None)
        rows = {}
        for r in names:
            corr = a * cur[r] + b
            rows[r] = dict(resid=relres(corr, real[r]), osc=osc(corr),
                           plateau=plateau(tmf, corr), plateau_real=plateau(tmf, real[r]),
                           fp_peak=firstpass_peak(tmf, corr), fp_peak_real=firstpass_peak(tmf, real[r]))
        methods[kind] = dict(a=float(a), b=float(b), rois=rows)
    out[Z] = dict(methods=methods, names=names)

# ---- tables ----
print("=== corrected per-ROI curve fidelity (relres after ONE global affine per method) ===")
print(f"{'slice':>5} {'method':>9} " + "".join(f"{r:>9}" for r in ROIS) + f"{'  scale a':>10}{'base b':>10}")
for Z in SLICES:
    for kind in ["CS", "NIK_R16", "NIK_full"]:
        m = out[Z]["methods"][kind]; cells = "".join(f"{m['rois'].get(r,{}).get('resid',float('nan')):>9.3f}" for r in ROIS)
        print(f"{Z:>5} {kind:>9} {cells}{m['a']:>10.3g}{m['b']:>10.2g}")

print("\n=== CS cortical-bias, RAW+affine (does CS cortex plateau sit ABOVE real?) ===")
for Z in SLICES:
    cs = out[Z]["methods"]["CS"]["rois"].get("cortex")
    if cs:
        rng = firstpass_peak(np.linspace(0,TA,len(mf)), np.ones(len(mf)))  # placeholder unused
        dev = (cs["plateau"] - cs["plateau_real"]) / (cs["plateau_real"] + 1e-9) * 100
        print(f"  sl{Z}: CS cortex plateau {cs['plateau']:.2e} vs real {cs['plateau_real']:.2e}  -> {dev:+.0f}% (old peak-norm claimed ~+12-20%)")

print("\n=== first-pass-peak blunting (1 - method_peak/real_peak), NIK_full vs CS ===")
for Z in SLICES:
    for roi in ["aorta", "cortex"]:
        line = f"  sl{Z} {roi:>7}: "
        for kind in ["NIK_full", "CS"]:
            rr = out[Z]["methods"][kind]["rois"].get(roi)
            if rr: line += f"{kind} {100*(1-rr['fp_peak']/(rr['fp_peak_real']+1e-9)):+5.0f}%  "
        print(line)

json.dump(out, open(f"{D}/task_R.json", "w"), indent=1, default=float)

# ---- figure: raw curves, each method after its OWN single global affine, vs real ----
fig, axes = plt.subplots(len(SLICES), len(ROIS), figsize=(4*len(ROIS), 3.1*len(SLICES)))
col = {"CS": "#08a", "NIK_R16": "#70c", "NIK_full": "#e62"}
for si, Z in enumerate(SLICES):
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; mf = np.load(f"{D}/step2_slice{Z}.npz")["mf"]; tmf = ctx["tmf"]
    names = out[Z]["names"]
    real = {r: np.array([im[rois[r]].mean() for im in mf]) for r in names}
    for r in ROIS:
        ax = axes[si, ROIS.index(r)]
        if r not in names: ax.axis("off"); continue
        ax.plot(tmf, real[r], "0.5", lw=1.6, label="real", zorder=5)
        for kind in ["CS", "NIK_R16", "NIK_full"]:
            m = out[Z]["methods"][kind]
            if kind == "CS": v = np.abs(np.load(f"{CSD}/cs_slice{Z:02d}_f25.npy")); tv = np.linspace(0,TA,v.shape[-1])
            else: v = np.abs(np.load(nikpath(kind, Z))); tv = np.linspace(0,TA,v.shape[-1])
            cur = np.interp(tmf, tv, np.array([v[..., i][rois[r]].mean() for i in range(v.shape[-1])]))
            ax.plot(tmf, m["a"]*cur + m["b"], col[kind], lw=1.1, label=kind, alpha=.85)
        ax.set_xlim(0, 260); ax.set_title(f"sl{Z} {r}", fontsize=9); ax.grid(alpha=.3)
        if si == 0 and r == names[0]: ax.legend(fontsize=6.5)
fig.suptitle("TASK R: RAW curves, each method after ONE global affine (fit across all ROIs) vs real", fontweight="bold")
fig.tight_layout(); p = fpath("taskR_raw_affine.png"); fig.savefig(p, dpi=130)
print(f"\nwrote {p.split('/')[-1]}")
