"""peak-frame recon panel for the STEP-1 temporal-expressiveness sweep vs existing models.
model-free NUFFT, input/output baselines, GRASP, and the sweep configs, each at its own aorta-peak frame."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RD = "results/realdata_nik_vs_cs_figures"
import os as _os
# reference-method plumbing. defaults = grasp-pro (unchanged). grasp v2:
#   CSD=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2 CSPRE=gv2 TAG=_gv2
_CSD = _os.environ.get("CSD", "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs")
_CSPRE = _os.environ.get("CSPRE", "cs"); _TAG = _os.environ.get("TAG", "")

ao = np.load("aif_slice21.npz")["ao"].astype(bool)
med = lambda v: np.median(v[ao], 0)
L = lambda p: np.abs(np.load(p)).astype(np.float32) if os.path.exists(p) else None
z = np.load("step2_slice21.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32)
items = [("model-free", mf),
         ("input-coil r16", L(f"{RD}/outcoil_subspace_input_slice21.npy")),
         ("output r16 w0-30", L(f"{RD}/outcoil_subspace_output_slice21.npy")),
         ("output r32 w0-30", L(f"{RD}/outcoil_subspace_output_r32_slice21.npy")),
         ("output r48 w0-30", L(f"{RD}/outcoil_subspace_output_r48_slice21.npy")),
         ("output r16 w0-90", L(f"{RD}/outcoil_subspace_output_pw90_slice21.npy")),
         ("output r32 w0-90", L(f"{RD}/outcoil_subspace_output_r32pw90_slice21.npy")),
         ("GRASP", L(f"{_CSD}/{_CSPRE}_slice21_f100.npy"))]
items = [(n, v) for n, v in items if v is not None]
ref = items[1][1]; m = ref.mean(-1) > ref.mean()*0.3; ys, xs = np.where(m)
y0, y1, x0, x1 = max(ys.min()-6, 0), ys.max()+6, max(xs.min()-6, 0), xs.max()+6
crop = lambda im: im[y0:y1, x0:x1]
n = len(items); fig, ax = plt.subplots(1, n, figsize=(2.6*n, 3.0))
for a, (nm, v) in zip(ax, items):
    pk = int(np.argmax(med(v))); im = crop(v[:, :, pk])
    a.imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im, 99.5)); a.set_title(nm, fontsize=9); a.axis("off")
fig.suptitle("real slice 21 aorta-peak frame: temporal-expressiveness sweep vs existing models", fontsize=11)
plt.tight_layout(); fig.savefig(f"{RD}/figures/step1_images{_TAG}.png", dpi=115, bbox_inches="tight"); print(f"SAVED figures/step1_images{_TAG}.png DONE_IMG")
