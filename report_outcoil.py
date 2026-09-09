"""Report: coil-as-input vs coil-as-output across NIK models (subspace/F0/F2) + GRASP/CS, real slice21.
Side-by-side recon images (peak frame) + contrast curves (aorta/cortex/medulla). Held-out NMSE table
parsed from the matched-run logs. NIK = 342 frames, CS = 122 frames -> curves plotted on seconds."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, glob, re, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RD = "/net/beegfs/users/P101440/DCE_NIK/results/realdata_nik_vs_cs_figures"
import os as _os
# reference-method plumbing. defaults = grasp-pro (unchanged). grasp v2:
#   CSD=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2 CSPRE=gv2 TAG=_gv2
_CSD = _os.environ.get("CSD", "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs")
_CSPRE = _os.environ.get("CSPRE", "cs"); _TAG = _os.environ.get("TAG", "")

TA = 375.0
ao = np.load("aif_slice21.npz")["ao"].astype(bool); kd = np.load("realkid_slice21.npz"); cx = kd["cortex"].astype(bool); md = kd["medulla"].astype(bool)
def med(v, m): return np.median(v[m], 0)
def norm0(c): return c/(np.median(c[:8])+1e-30)
def load(p): return np.abs(np.load(p)).astype(np.float32) if os.path.exists(p) else None
models = ["subspace", "f0", "f2"]; modes = ["input", "output"]
EXIST_IN = {"subspace": "results_spoke_full_slice21/nik_slice_21.npy", "f0": "results_batch/pk_f0_sl21/nik_slice_21.npy", "f2": "results_batch/pk_f2_sl21/nik_slice_21.npy"}
rec = {}; lbl = {}
for m in models:
    for md_ in modes:
        p = f"{RD}/outcoil_{m}_{md_}_slice21.npy"; tag = "matched"
        if md_ == "input" and not os.path.exists(p): p = EXIST_IN.get(m, ""); tag = "existing"
        r = load(p)
        if r is not None: rec[(m, md_)] = r; lbl[(m, md_)] = tag
grasp = load(f"{_CSD}/{_CSPRE}_slice21_f100.npy")
mfz = np.load("step2_slice21.npz"); mf = np.abs(mfz["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(mfz["tmf"])  # model-free NUFFT ref [192,192,240]
# held-out NMSE from logs
nmse = {}
for f in glob.glob(f"{RD}/ocreal_*.log"):
    for line in open(f):
        mt = re.search(r"REAL-(\w+)-(INPUT|OUTPUT) slice21: TEST held-out NMSE ([0-9.e+-]+)", line)
        if mt: nmse[(mt.group(1), mt.group(2).lower())] = float(mt.group(3))
print("held-out NMSE:", {k: round(v, 4) for k, v in nmse.items()})

# peak frame from a subspace recon
ref = rec.get(("subspace", "output")); ref = ref if ref is not None else next(iter(rec.values())); pk = int(np.argmax(med(ref, ao)))
mask = ref.mean(-1) > ref.mean()*0.3; ys, xs = np.where(mask); y0, y1, x0, x1 = max(ys.min()-6, 0), ys.max()+6, max(xs.min()-6, 0), xs.max()+6
crop = lambda im: im[y0:y1, x0:x1]
nrow = len(models); fig = plt.figure(figsize=(15, 4.2*nrow+5))
gs = fig.add_gridspec(nrow+1, 4, height_ratios=[1]*nrow+[1.1])
for i, m in enumerate(models):
    for j, md_ in enumerate(modes):
        ax = fig.add_subplot(gs[i, j]); r = rec.get((m, md_))
        if r is not None:
            vmax = float(np.percentile(r[:, :, pk], 99.5)); ax.imshow(crop(r[:, :, pk]), cmap="gray", vmin=0, vmax=vmax)
            tag = f"{m} / {md_}-coil [{lbl.get((m, md_), '')}]"; n = nmse.get((m, md_)); tag += f"  NMSE {n:.3f}" if n else ""
            ax.set_title(tag, fontsize=10)
        ax.axis("off")
    # GRASP in col 2, difference input-vs-output in col 3
    axg = fig.add_subplot(gs[i, 2])
    if grasp is not None and i == 0:
        vmg = float(np.percentile(grasp[:, :, np.argmax(med(grasp, ao))], 99.5)); axg.imshow(crop(grasp[:, :, np.argmax(med(grasp, ao))]), cmap="gray", vmin=0, vmax=vmg); axg.set_title("GRASP/CS (f100)", fontsize=10)
    axg.axis("off")
    axd = fig.add_subplot(gs[i, 3])
    if (m, "input") in rec and (m, "output") in rec:
        a, b = rec[(m, "input")][:, :, pk], rec[(m, "output")][:, :, pk]
        axd.imshow(crop(np.abs(a/a.max()-b/b.max())), cmap="magma", vmin=0, vmax=0.3); axd.set_title(f"{m}: |in-out| x1", fontsize=9)
    axd.axis("off")
# curves row
tN = np.linspace(0, TA, ref.shape[2]); tG = np.linspace(0, TA, grasp.shape[2]) if grasp is not None else None
for j, (nm, mk) in enumerate([("aorta", ao), ("cortex", cx), ("medulla", md)]):
    ax = fig.add_subplot(gs[nrow, j])
    ax.plot(tmf, norm0(med(mf, mk)), color="0.45", lw=3.0, alpha=0.85, zorder=1, label="model-free (NUFFT)")  # reference
    for m in models:
        for md_, ls in [("input", "--"), ("output", "-")]:
            if (m, md_) in rec: ax.plot(tN, norm0(med(rec[(m, md_)], mk)), ls, lw=1.1, zorder=3, label=f"{m}-{md_}")
    if grasp is not None: ax.plot(tG, norm0(med(grasp, mk)), "k:", lw=1.6, zorder=2, label="GRASP")
    ax.set_title(nm, fontsize=10); ax.set_xlabel("time (s)")
    if j == 0: ax.legend(fontsize=6, ncol=2)
fig.suptitle("real slice21: coil input vs output x {subspace,F0,F2} + GRASP", fontsize=13)
plt.tight_layout(); fig.savefig(f"{RD}/figures/outcoil_report{_TAG}.png", dpi=115); print(f"SAVED figures/outcoil_report{_TAG}.png")
print("DONE_REPORT")
