"""Spoke-reduction frontier, judged by EYE. NIK vs CS on IDENTICAL acquired spokes,
descending spokes/frame. Rows = fraction, cols = [NIK mean | CS mean | NIK frame | CS frame].
Look for the crossover: where does CS break into streaks while NIK holds structure?
out: spoke_frontier_sl13.png"""
import numpy as np, os, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from figpath import fig as fpath
D = "/scratch/rnga/vvpshenov/DCE_NIK"; CS = "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs"
SL = 13
LABS = [("f100", "100% (14 sp/fr)"), ("f70", "71% (10)"), ("f50", "50% (7)"),
        ("f35", "36% (5)"), ("f25", "29% (4)")]

def load_nik(lab):
    p = f"{D}/results_spoke_nik_{lab}/nik_slice_{SL:02d}.npy"
    return np.abs(np.load(p)) if os.path.exists(p) else None
def load_cs(lab):
    p = f"{CS}/cs_slice{SL:02d}_{lab}.npy"
    return np.abs(np.load(p)) if os.path.exists(p) else None

rows = [(lab, name, load_nik(lab), load_cs(lab)) for lab, name in LABS]
rows = [r for r in rows if r[2] is not None or r[3] is not None]
if not rows:
    print("no results yet"); raise SystemExit

# arterial-ish frame index picked from the full-CS bolus (peak of aorta-region mean)
ref = load_cs("f100")
fr_cs = ref.shape[-1] // 5 if ref is not None else 0
nref = load_nik("f100")
fr_nik = nref.shape[-1] // 5 if nref is not None else 0

fig, ax = plt.subplots(len(rows), 4, figsize=(13, 3.05 * len(rows)))
if len(rows) == 1: ax = ax[None, :]
col_titles = ["NIK — temporal mean", "CS — temporal mean", "NIK — arterial frame", "CS — arterial frame"]
def show(a, img, vmax, title=None):
    a.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=vmax); a.axis("off")
    if title: a.set_title(title, fontsize=10)
for i, (lab, name, nik, cs) in enumerate(rows):
    vmn = np.percentile(nik.mean(-1), 99.5) if nik is not None else 1
    vmc = np.percentile(cs.mean(-1), 99.5) if cs is not None else 1
    if nik is not None: show(ax[i, 0], nik.mean(-1), vmn, col_titles[0] if i == 0 else None)
    else: ax[i, 0].axis("off")
    if cs is not None: show(ax[i, 1], cs.mean(-1), vmc, col_titles[1] if i == 0 else None)
    else: ax[i, 1].axis("off")
    if nik is not None: show(ax[i, 2], nik[..., min(fr_nik, nik.shape[-1]-1)], np.percentile(nik, 99.5), col_titles[2] if i == 0 else None)
    else: ax[i, 2].axis("off")
    if cs is not None: show(ax[i, 3], cs[..., min(fr_cs, cs.shape[-1]-1)], np.percentile(cs, 99.5), col_titles[3] if i == 0 else None)
    else: ax[i, 3].axis("off")
    ax[i, 0].text(-0.12, 0.5, name, rotation=90, va="center", ha="center",
                  transform=ax[i, 0].transAxes, fontsize=11, fontweight="bold")
fig.suptitle(f"Spoke-reduction frontier — NIK vs CS, identical spokes, slice {SL}\n(where does CS streak while NIK holds?)",
             fontweight="bold", fontsize=12)
fig.tight_layout(); fig.savefig(fpath(f"spoke_frontier_sl{SL}.png"), dpi=135, bbox_inches="tight")
print(f"rows present: {[r[0] for r in rows]} -> spoke_frontier_sl{SL}.png")
