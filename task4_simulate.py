"""TASK 4 Stage 1: motion-free matched-model XCAT simulation. Truth dynamic follows EXACTLY the
F0 basis I(x,t)=theta_AIF*AIF(t)+theta_intAIF*intAIF(t)+theta_baseline. Forward via cs_nufft's
own SENSE-NUFFT operator (so the direct-discrete recon is forward-identical). Fixed noise.
Saves truth maps/dynamic, k-space, input/non-input masks, timestamps, traj, coils, ROIs, basis."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os, sys, json, csv, scipy.ndimage as ndi
sys.path.insert(0, "."); sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
import xcat_adapter as X
from cs_nufft import NufftSubspace
D = "/net/beegfs/users/P101440/DCE_NIK"; OUT = f"{D}/results/task4_xcat_nomotion_pilot"
for s in ["figures", "arrays", "logs", "checkpoints", "script_snapshot"]: os.makedirs(f"{OUT}/{s}", exist_ok=True)
ZI = 5; N = 220; AORTA_LABEL = 36; F25_KEEP_ANGLES = 2; NOISE_REL = 0.02; SEED = 20260813

# ---- geometry from XCAT slice zi (coils, traj, times) ----
d = X.slice_radial(ZI); kx, ky, times = d["kx"], d["ky"], d["times"]      # kx/ky [F,NA,RO] in [-0.5,0.5]; times [F] s
F, NA, RO = kx.shape; b1 = np.transpose(d["coil"], (1, 2, 0)).astype(np.complex64)  # [220,220,8]
C = b1.shape[-1]; Tt = float(times.max())
dce_full, traj_full, coil_full, tim_full, lab_all, ref_all = X._load()
lab = np.array(lab_all)[ZI]; ref0 = np.abs(np.array(ref_all)[0, ZI])       # anatomy label [152,220]; GT pre-contrast img
# embed 152x220 label + ref into 220x220 (center), matching aorta_roi convention
def embed(a):
    out = np.zeros((N, N), a.dtype); h, w = a.shape; y0 = (N - h) // 2; x0 = (N - w) // 2; out[y0:y0 + h, x0:x0 + w] = a; return out
labE = embed(lab.astype(int)) if lab.shape != (N, N) else lab.astype(int)
ref0E = embed(ref0) if ref0.shape != (N, N) else ref0
if ref0E.T.shape == labE.shape and ref0E.shape != labE.shape: ref0E = ref0E.T

# ---- F0 temporal basis on XCAT times (matched to model via aif_xcat.npz) ----
az = np.load(f"{D}/aif_slice21.npz"); realtc = np.asarray(az["tC"]); realaif = np.asarray(az["aif_frame"])
aif_x = np.interp(times, realtc / realtc[-1] * Tt, realaif)                # real AIF shape resampled onto XCAT times
np.savez(f"{D}/aif_xcat.npz", aif_frame=aif_x.astype(np.float32), tC=times.astype(np.float32), verdict="matched-model", ttp=float(times[np.argmax(aif_x)]))
aifn = aif_x / (aif_x.max() + 1e-9)                                        # peak-norm  (col 0)
integ = np.concatenate([[0], np.cumsum(0.5 * (aifn[1:] + aifn[:-1]) * np.diff(times))]); integn = integ / (integ.max() + 1e-9)  # max-norm (col1)
Phi = np.stack([aifn, integn, np.ones_like(aifn)], 1).astype(np.complex64)  # [F,3]

# ---- TRUTH coefficient maps (defined from anatomy BEFORE any reconstruction) ----
body = labE > 0; aorta = labE == AORTA_LABEL
th_baseline = (ref0E / (np.percentile(ref0E[body], 99) + 1e-9)).clip(0, 1) * body   # anatomical baseline
th_AIF = aorta.astype(np.float32) * 1.0                                             # small vessel (high-freq)
th_intAIF = (body & ~aorta).astype(np.float32) * 1.0                                # broad tissue (smooth)
theta_true = np.stack([th_AIF, th_intAIF, th_baseline], -1).astype(np.complex64)    # [220,220,3]

# ---- ROIs (from anatomy; cortex/medulla = two largest non-aorta organ labels as proxies) ----
areas = {int(l): int((labE == l).sum()) for l in np.unique(labE) if l not in (0, AORTA_LABEL)}
organL = sorted(areas, key=areas.get, reverse=True)
cortex_lab, medulla_lab = organL[0], organL[1]
ROI = dict(aorta=aorta, cortex=(labE == cortex_lab), medulla=(labE == medulla_lab),
           organ=(body & ~aorta), full=body)
np.savez(f"{OUT}/arrays/rois.npz", **{k: v for k, v in ROI.items()})

# ---- dynamic truth series ----
I_true = np.tensordot(theta_true, Phi.T, axes=([2], [0]))                  # [220,220,F] complex

# ---- forward with cs_nufft operator (all 9 angles = f100) ----
trajs_full = [(kx[t], ky[t]) for t in range(F)]
E_full = NufftSubspace(Phi, b1, trajs_full, N)
y_clean = E_full.fwd(theta_true)                                          # list F of [C, NA*RO]
rng = np.random.default_rng(SEED)
sig = np.mean([np.abs(yt).mean() for yt in y_clean])                       # noise scaled to mean |y|
y_noisy = [yt + NOISE_REL * sig * (rng.standard_normal(yt.shape) + 1j * rng.standard_normal(yt.shape)) for yt in y_clean]

# ---- input / non-input masks (f100 all; f25 keep first 2 angles/frame) ----
keep = np.zeros((F, NA), bool); keep[:, :F25_KEEP_ANGLES] = True           # deterministic keep rule
# reshape per-frame samples [C, NA*RO] -> [C, NA, RO] to split
def split(yl, mask_keep):
    kept, non = [], []
    for t in range(F):
        yt = yl[t].reshape(C, NA, RO)
        kept.append(yt[:, mask_keep[t], :].reshape(C, -1)); non.append(yt[:, ~mask_keep[t], :].reshape(C, -1))
    return kept, non
keep_f100 = np.ones((F, NA), bool)
y100, _ = split(y_noisy, keep_f100)
y25_in, y25_non = split(y_noisy, keep)
trajs_f25 = [(kx[t][keep[t]], ky[t][keep[t]]) for t in range(F)]
trajs_f25_non = [(kx[t][~keep[t]], ky[t][~keep[t]]) for t in range(F)]

# ---- save everything ----
np.savez(f"{OUT}/arrays/sim.npz", theta_true=theta_true, I_true=I_true.astype(np.complex64), Phi=Phi,
         b1=b1, times=times.astype(np.float32), kx=kx.astype(np.float32), ky=ky.astype(np.float32),
         keep_f25=keep, labels=labE, aif_frame=aif_x.astype(np.float32),
         cortex_lab=cortex_lab, medulla_lab=medulla_lab)
np.save(f"{OUT}/arrays/y100.npy", np.array(y100, dtype=object), allow_pickle=True)
np.save(f"{OUT}/arrays/y25_in.npy", np.array(y25_in, dtype=object), allow_pickle=True)
np.save(f"{OUT}/arrays/y25_non.npy", np.array(y25_non, dtype=object), allow_pickle=True)

# ---- Part 10 sanity: noiseless f100 recovery of truth via the SAME operator (adjoint + direct) ----
y_clean100, _ = split(y_clean, keep_f100)
adj = E_full.adj(y_clean100); adj /= (np.abs(adj).max() + 1e-12)
# quick 15-iter direct recon of NOISELESS f100 (should recover theta well -> validates forward)
from cs_nufft import recon as cs_recon
th_f100_noiseless = cs_recon(E_full, y_clean100, iters=25, ls=1e-4, lt=1e-4, verbose=False)
def nrmse(a, b, m): return float(np.linalg.norm(a[m] - b[m]) / (np.linalg.norm(b[m]) + 1e-12))
# scale-match (global complex) then compare intAIF & baseline (AIF is tiny/vessel)
s = np.vdot(th_f100_noiseless[body], theta_true[body]) / (np.vdot(th_f100_noiseless[body], th_f100_noiseless[body]) + 1e-12)
thc = th_f100_noiseless * s
san = dict(f100_noiseless_intAIF_NRMSE=nrmse(np.abs(thc[..., 1]), np.abs(theta_true[..., 1]), body),
           f100_noiseless_baseline_NRMSE=nrmse(np.abs(thc[..., 2]), np.abs(theta_true[..., 2]), body),
           f100_noiseless_AIF_NRMSE=nrmse(np.abs(thc[..., 0]), np.abs(theta_true[..., 0]), aorta if aorta.sum() else body),
           noise_rel=NOISE_REL, F=F, NA=NA, RO=RO, C=C, Tt=Tt, N=N, zi=ZI,
           keep_f25_angles=F25_KEEP_ANGLES, cortex_lab=int(cortex_lab), medulla_lab=int(medulla_lab),
           aorta_vox=int(aorta.sum()), body_vox=int(body.sum()))
json.dump(san, open(f"{OUT}/arrays/sim_meta.json", "w"), indent=1, default=float)
print("SIM built. F=%d NA=%d RO=%d C=%d Tt=%.1fs | aorta %d vox | cortex_lab %d medulla_lab %d" % (F, NA, RO, C, Tt, aorta.sum(), cortex_lab, medulla_lab))
print("f100 NOISELESS forward-verify (direct recon vs truth): intAIF NRMSE %.3f | baseline %.3f | AIF %.3f" % (
    san["f100_noiseless_intAIF_NRMSE"], san["f100_noiseless_baseline_NRMSE"], san["f100_noiseless_AIF_NRMSE"]))
print("wrote", OUT)
