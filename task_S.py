"""TASK S: is CS's oscillation NOISE or PHYSIOLOGY? decides the NIK denoising claim.
S1 reference oscillation (model-free = closest to truth) + CS-f100-vs-f25 persistence test
   (if CS oscillation vanishes with ALL spokes -> undersampling noise; if it persists -> signal).
S2 spectral character of the residual (method - smooth), broadband=noise / narrowband(resp 0.2-0.3Hz)=physiology.
S3 ROI pattern (aorta-specific narrowband = pulsatility; aorta-specific broadband = noisier recon).
S4 liver residual in ABSOLUTE terms (relres inflates small-denominator liver).
ROIs aorta/cortex/medulla/liver, slices 18/19/21. out: task_S.json + spectra figure."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, sys
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, ".")
from figpath import fig as fpath
import consolidated as C
D = "/net/beegfs/users/P101440/DCE_NIK"; REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"
import os
CSD = os.environ.get("CSD", "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs")
CSPRE = os.environ.get("CSPRE", "cs"); TAG = os.environ.get("TAG", ""); TA = 375.0
SLICES = [18, 19, 21]; ROIS = ["aorta", "cortex", "medulla", "liver"]
def nfull(Z): return f"{D}/results_batch/full_sl{Z}{'f25' if Z==21 else ''}/nik_slice_{Z}_cplx.npy"

def osc(c): sm = savgol_filter(c, 11, 3); return float(np.std(c - sm) / (abs(sm).max() + 1e-9))
def resid_hf(c):                                          # high-freq residual = curve minus savgol trend
    return c - savgol_filter(c, 11, 3)
def spectrum(c, t):                                       # power spectrum of the HF residual on a uniform grid
    tu = np.linspace(t.min(), t.max(), len(t)); cu = np.interp(tu, t, resid_hf(c)); dt = tu[1] - tu[0]
    w = np.hanning(len(cu)); f = np.fft.rfftfreq(len(cu), dt); P = np.abs(np.fft.rfft(cu * w)) ** 2
    return f, P
def flatness(P):                                          # spectral flatness: ~1 broadband/white, ~0 tonal/narrowband
    P = P[1:] + 1e-20; return float(np.exp(np.mean(np.log(P))) / np.mean(P))
def resp_frac(f, P):                                      # fraction of HF power in the respiratory band 0.2-0.35 Hz
    band = (f >= 0.2) & (f <= 0.35); return float(P[band].sum() / (P[1:].sum() + 1e-20))

out = {}
for Z in SLICES:
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; mf = np.load(f"{D}/step2_slice{Z}.npz")["mf"]; tmf = ctx["tmf"]
    names = [r for r in ROIS if rois.get(r) is not None and rois[r].sum() > 0]
    curves = {}                                           # method -> (t, {roi: curve})
    curves["ref"] = (tmf, {r: np.array([im[rois[r]].mean() for im in mf]) for r in names})
    cs100 = ctx.get("cs_meth") if ctx.get("cs_meth") is not None else ctx["cs100"]; curves["CS_f100"] = (np.linspace(0, TA, cs100.shape[-1]),
                                               {r: np.array([cs100[..., i][rois[r]].mean() for i in range(cs100.shape[-1])]) for r in names})
    cs25 = np.abs(np.load(f"{CSD}/{CSPRE}_slice{Z:02d}_f25.npy")).astype(np.float32); tC25 = np.linspace(0, TA, cs25.shape[-1])
    curves["CS_f25"] = (tC25, {r: np.array([cs25[..., i][rois[r]].mean() for i in range(cs25.shape[-1])]) for r in names})
    nf = np.abs(np.load(nfull(Z))).astype(np.float32); tN = np.linspace(0, TA, nf.shape[-1])
    curves["NIK_full"] = (tN, {r: np.array([nf[..., i][rois[r]].mean() for i in range(nf.shape[-1])]) for r in names})
    rec = {}
    for meth, (t, cs) in curves.items():
        rec[meth] = {}
        for r in names:
            f, P = spectrum(cs[r], t)
            rec[meth][r] = dict(osc=osc(cs[r]), flat=flatness(P), respfrac=resp_frac(f, P),
                                abs_resid_rms=float(np.std(resid_hf(cs[r]))))   # absolute (unnormalized) HF residual
    out[Z] = rec

# S1 + S3 table
print("=== S1/S3 oscillation per ROI (osc), and the CS f100-vs-f25 persistence test ===")
print(f"{'slice':>5} {'method':>9}" + "".join(f"{r:>9}" for r in ROIS))
for Z in SLICES:
    for meth in ["ref", "CS_f100", "CS_f25", "NIK_full"]:
        print(f"{Z:>5} {meth:>9}" + "".join(f"{out[Z][meth].get(r,{}).get('osc',float('nan')):>9.3f}" for r in ROIS))
print("READ S1: if CS_f100 osc << CS_f25 -> CS oscillation is undersampling/recon NOISE (denoising claim holds).")
print("        if ref osc is also high -> real fluctuation NIK erases (claim dies).")

# S2 spectral flatness (broadband vs narrowband) + respiratory fraction, CS_f100 (best resolved) and CS_f25
print("\n=== S2 spectral character of CS residual (flat~1=broadband/noise, ~0=tonal; respfrac=power in 0.2-0.35Hz) ===")
print(f"{'slice':>5} {'method':>9}" + "".join(f"{r+'_flat':>13}" for r in ['aorta','cortex']) + f"{'aorta_respfrac':>15}")
for Z in SLICES:
    for meth in ["CS_f100", "CS_f25"]:
        a = out[Z][meth]; print(f"{Z:>5} {meth:>9}{a.get('aorta',{}).get('flat',float('nan')):>13.2f}{a.get('cortex',{}).get('flat',float('nan')):>13.2f}{a.get('aorta',{}).get('respfrac',float('nan')):>15.3f}")

# S4 liver absolute vs relative
print("\n=== S4 liver residual: ABSOLUTE HF-resid RMS (relres inflated liver via small denominator) ===")
for Z in SLICES:
    print(f"  sl{Z}: " + "  ".join(f"{m} {out[Z][m].get('liver',{}).get('abs_resid_rms',float('nan')):.2e}" for m in ["ref","CS_f100","CS_f25","NIK_full"]))

json.dump(out, open(f"{D}/task_S{TAG}.json", "w"), indent=1, default=float)

# ---- spectra figure: HF-residual power spectrum per ROI, CS_f100 vs CS_f25 vs NIK vs ref ----
fig, axes = plt.subplots(len(SLICES), len(ROIS), figsize=(4*len(ROIS), 3.1*len(SLICES)))
col = {"ref": "0.5", "CS_f100": "#088", "CS_f25": "#08a", "NIK_full": "#e62"}
for si, Z in enumerate(SLICES):
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; mf = np.load(f"{D}/step2_slice{Z}.npz")["mf"]; tmf = ctx["tmf"]
    names = [r for r in ROIS if rois.get(r) is not None and rois[r].sum() > 0]
    cs100 = ctx.get("cs_meth") if ctx.get("cs_meth") is not None else ctx["cs100"]; cs25 = np.abs(np.load(f"{CSD}/{CSPRE}_slice{Z:02d}_f25.npy")); nf = np.abs(np.load(nfull(Z)))
    src = {"ref": (tmf, mf, None), "CS_f100": (np.linspace(0,TA,cs100.shape[-1]), cs100, None),
           "CS_f25": (np.linspace(0,TA,cs25.shape[-1]), cs25, None), "NIK_full": (np.linspace(0,TA,nf.shape[-1]), nf, None)}
    for r in ROIS:
        ax = axes[si, ROIS.index(r)]
        if r not in names: ax.axis("off"); continue
        for meth, (t, vol, _) in src.items():
            if meth == "ref": cur = np.array([im[rois[r]].mean() for im in vol])
            else: cur = np.array([vol[..., i][rois[r]].mean() for i in range(vol.shape[-1])])
            cur = cur / (savgol_filter(cur, 11, 3).max() + 1e-12)          # normalize so spectra comparable
            f, P = spectrum(cur, t); ax.semilogy(f, P + 1e-12, col[meth], lw=1, label=meth)
        ax.axvspan(0.2, 0.35, color="g", alpha=.08)                        # respiratory band
        ax.set_xlim(0, 0.45); ax.set_title(f"sl{Z} {r}", fontsize=9); ax.grid(alpha=.3)
        if si == 0 and r == names[0]: ax.legend(fontsize=6)
        if si == len(SLICES)-1: ax.set_xlabel("Hz", fontsize=8)
fig.suptitle("TASK S: HF-residual power spectra (green=respiratory 0.2-0.35Hz). broadband=noise, narrowband peak=physiology", fontweight="bold")
fig.tight_layout(); p = fpath(f"taskS_spectra{TAG}.png"); fig.savefig(p, dpi=130); print(f"\nwrote {p.split('/')[-1]}")
