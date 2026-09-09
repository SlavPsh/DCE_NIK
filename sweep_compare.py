"""Comparative figure for the temporal-expressiveness experiments (rank / phi_w0 / activation) vs the
models we already have: input-coil, baseline output-coil, GRASP, model-free NUFFT reference. Auto-
discovers outcoil_subspace_output*_slice21.npy variants. Metrics: aorta-peak/model-free ratio (bolus
recovery), curve NMSE vs model-free (aorta/cortex/medulla), + held-out NMSE parsed from logs. Saves a
figure (images at peak + curves) for the NIK-vs-CS notebook. usage: python sweep_compare.py [outtag]"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, glob, re, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RD = "results/realdata_nik_vs_cs_figures"; TA = 375.0; OUT = sys.argv[1] if len(sys.argv) > 1 else "sweep"
import os as _os
# reference-method plumbing. defaults = grasp-pro (unchanged). grasp v2:
#   CSD=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2 CSPRE=gv2 TAG=_gv2
_CSD = _os.environ.get("CSD", "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs")
_CSPRE = _os.environ.get("CSPRE", "cs"); _TAG = _os.environ.get("TAG", "")

ao = np.load("aif_slice21.npz")["ao"].astype(bool); kd = np.load("realkid_slice21.npz"); cx = kd["cortex"].astype(bool); md = kd["medulla"].astype(bool)
z = np.load("step2_slice21.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"])
med = lambda v, m: np.median(v[m], 0); n0 = lambda c: c/(np.median(c[:8])+1e-30); load = lambda p: np.abs(np.load(p)).astype(np.float32) if os.path.exists(p) else None
# held-out NMSE per (rank,phi_w0,t_sigma,coil) from all ocreal/ocrsw logs
nm_by_cfg = {}
for f in glob.glob(f"{RD}/ocr*.log"):
    txt = open(f).read(); cfg = re.search(r"model=subspace coil=(\w+) rank=(\d+) phi_w0=([\d.]+) t_sigma=([\d.]+)", txt); tn = re.search(r"TEST held-out NMSE ([0-9.e+-]+)", txt)
    if cfg and tn: nm_by_cfg[(cfg.group(1), int(cfg.group(2)), float(cfg.group(3)), float(cfg.group(4)))] = float(tn.group(1))
# assemble configs: label -> (recon, time, style, heldNMSE)
tN = np.linspace(0, TA, 342); tG = np.linspace(0, TA, 122); cfgs = []
inp = load(f"{RD}/outcoil_subspace_input_slice21.npy")
if inp is not None: cfgs.append(("input-coil r16", inp, tN, dict(ls="--", c="C0", lw=1.2), nm_by_cfg.get(("input", 16, 30.0, 0.0))))
for p in sorted(glob.glob(f"{RD}/outcoil_subspace_output*_slice21.npy")):
    tag = re.search(r"output(.*?)_slice21", p).group(1)  # '' baseline, '_r32','_pw90',...
    lab = "output-coil r16" if tag == "" else "output" + tag.replace("_", " ")
    rm = re.search(r"r(\d+)", tag); r = int(rm.group(1)) if rm else 16                                   # safe parse (activation tags have no r/pw digits)
    wm = re.search(r"pw(\d+)", tag); w = float(wm.group(1)) if wm else 30.0
    cfgs.append((lab, load(p), tN, dict(ls="-", lw=1.2), nm_by_cfg.get(("output", r, w, 0.0))))
grasp = load(f"{_CSD}/{_CSPRE}_slice21_f100.npy")
if grasp is not None: cfgs.append(("GRASP f100 (100%)", grasp, tG, dict(ls=":", c="k", lw=1.6), None))
g80 = load(f"{_CSD}/{_CSPRE}_slice21_f80match.npy")
if g80 is not None: cfgs.append(("GRASP f80-match (80%)", g80, tG, dict(ls=":", c="C3", lw=1.6), None))  # same spokes as nik
# metrics table
print(f"{'config':22s} {'aortaPk/mf':>10s} {'NMSEmf_ao':>9s} {'NMSEmf_cx':>9s} {'NMSEmf_md':>9s} {'heldNMSE':>9s}")
mfpk = {nm: n0(med(mf, mk)).max() for nm, mk in [("aorta", ao)]}
def cnmse(v, tv, mk): ci = np.interp(tmf, tv, n0(med(v, mk))); r = n0(med(mf, mk)); return float(np.linalg.norm(ci-r)/(np.linalg.norm(r)+1e-9))
rows = []
for lab, v, tv, st, hn in cfgs:
    if v is None: continue
    pk = n0(med(v, ao)).max()/mfpk["aorta"]; na, nc, nmd = cnmse(v, tv, ao), cnmse(v, tv, cx), cnmse(v, tv, md)
    print(f"{lab:22s} {pk:10.2f} {na:9.3f} {nc:9.3f} {nmd:9.3f} {('%.3f'%hn) if hn else '-':>9s}"); rows.append((lab, pk, na, nc, nmd, hn))
# figure: curves (3 ROIs) with model-free reference thick grey
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for j, (nm, mk) in enumerate([("aorta", ao), ("cortex", cx), ("medulla", md)]):
    ax[j].plot(tmf, n0(med(mf, mk)), color="0.45", lw=3.2, alpha=.85, zorder=1, label="model-free (NUFFT)")
    for lab, v, tv, st, hn in cfgs:
        if v is not None: ax[j].plot(tv, n0(med(v, mk)), label=lab, zorder=3, **st)
    ax[j].set_title(nm, fontsize=11); ax[j].set_xlabel("time (s)")
ax[0].legend(fontsize=7, ncol=2)
fig.suptitle("temporal expressiveness sweep vs existing models (real slice 21, output-coil subspace)", fontsize=12)
plt.tight_layout(); fig.savefig(f"{RD}/figures/{OUT}_compare{_TAG}.png", dpi=120); print(f"SAVED figures/{OUT}_compare{_TAG}.png"); print("DONE_SWCMP")
