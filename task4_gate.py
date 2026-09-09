"""TASK 4 gate: forward-model verification (cheap + decisive). (1) adjoint dot-product test:
<E x, y> == <x, E^H y>  -> proves fwd/adj are true transposes (no sign/convention bug).
(2) density-compensated adjoint of NOISELESS f100 -> project onto basis -> correlation/NRMSE vs
KNOWN truth (structural geometry+basis check, PSF-limited). (3) short bounded direct recon
(per-iter prints) as a convergence sanity. Validates basis+forward+solver before any method."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys, json, time; sys.path.insert(0, ".")
from cs_nufft import NufftSubspace, recon as cs_recon
OUT = "/net/beegfs/users/P101440/DCE_NIK/results/task4_xcat_nomotion_pilot"
S = np.load(f"{OUT}/arrays/sim.npz"); Phi = S["Phi"]; b1 = S["b1"]; kx = S["kx"]; ky = S["ky"]
F, NA, RO = kx.shape; C = b1.shape[-1]; N = RO; th = S["theta_true"]; labels = S["labels"]
body = labels > 0; aorta = labels == 36
def nrmse(a, b, m): return float(np.linalg.norm(a[m] - b[m]) / (np.linalg.norm(b[m]) + 1e-12))
def corr(a, b, m): a = a[m].ravel(); b = b[m].ravel(); return float(np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
trajs100 = [(kx[t], ky[t]) for t in range(F)]
E = NufftSubspace(Phi, b1, trajs100, N)

# (1) adjoint dot-product test
rng = np.random.default_rng(0)
xr = (rng.standard_normal(th.shape) + 1j * rng.standard_normal(th.shape)).astype(np.complex64)
t0 = time.time(); Ex = E.fwd(xr); print(f"[adjtest] fwd {time.time()-t0:.0f}s", flush=True)
yr = [(rng.standard_normal(np.asarray(e).shape) + 1j * rng.standard_normal(np.asarray(e).shape)).astype(np.complex64) for e in Ex]
Hy = E.adj(yr)
lhs = sum(np.vdot(np.asarray(Ex[t]).ravel(), np.asarray(yr[t]).ravel()) for t in range(F))
rhs = np.vdot(xr.ravel(), Hy.ravel())
adj_rel = float(np.abs(lhs - rhs) / (np.abs(lhs) + 1e-12))
print(f"[adjtest] |<Ex,y>-<x,E^Hy>|/|.| = {adj_rel:.2e}", flush=True)

# (2) density-compensated adjoint of noiseless f100 -> project onto Phi
t0 = time.time(); y_clean = E.fwd(th); print(f"[adj] fwd noiseless {time.time()-t0:.0f}s", flush=True)
dcf = [np.maximum(np.abs(kx[t] + 1j * ky[t]), 1e-3).reshape(-1) for t in range(F)]
yi = [np.asarray(y_clean[t]) * np.tile(dcf[t], (C, 1)) for t in range(F)]
adjx = E.adj(yi)                                   # coefficient-domain adjoint (already SENSE+basis)
adj_metrics = {f"adj_{cn}_corr": corr(np.abs(adjx[..., i]), np.abs(th[..., i]), aorta if cn == "AIF" else body)
               for i, cn in enumerate(["AIF", "intAIF", "baseline"])}
print("[adj] proj corr vs truth:", {k: round(v, 3) for k, v in adj_metrics.items()}, flush=True)

# (3) short bounded recon (per-iter prints), scale-matched NRMSE vs truth
class V:                                            # force per-iter verbose
    pass
t0 = time.time(); thd = cs_recon(E, y_clean, iters=8, ls=1e-4, lt=1e-4, verbose=True); print(f"[recon] {time.time()-t0:.0f}s", flush=True)
s = np.vdot(thd[body], th[body]) / (np.vdot(thd[body], thd[body]) + 1e-12); thc = thd * s
san = dict(adjoint_dotproduct_rel=adj_rel, **adj_metrics,
           recon8_intAIF_NRMSE=nrmse(np.abs(thc[..., 1]), np.abs(th[..., 1]), body),
           recon8_baseline_NRMSE=nrmse(np.abs(thc[..., 2]), np.abs(th[..., 2]), body),
           recon8_AIF_NRMSE=nrmse(np.abs(thc[..., 0]), np.abs(th[..., 0]), aorta),
           recon8_complex_NRMSE=float(np.linalg.norm((thc - th)[body]) / (np.linalg.norm(th[body]) + 1e-12)),
           noise_rel=0.02, F=int(F), NA=int(NA), RO=int(RO), C=int(C), Tt=float(S["times"].max()), N=int(N), zi=5,
           keep_f25_angles=2, cortex_lab=int(S["cortex_lab"]), medulla_lab=int(S["medulla_lab"]),
           aorta_vox=int(aorta.sum()), body_vox=int(body.sum()), seed=20260813, recon_iters=8)
json.dump(san, open(f"{OUT}/arrays/sim_meta.json", "w"), indent=1, default=float)
print("GATE adj_rel %.1e | adj_corr intAIF %.3f baseline %.3f | recon8 intAIF %.3f baseline %.3f AIF %.3f" % (
    adj_rel, san["adj_intAIF_corr"], san["adj_baseline_corr"], san["recon8_intAIF_NRMSE"], san["recon8_baseline_NRMSE"], san["recon8_AIF_NRMSE"]), flush=True)
print("GATE_DONE", flush=True)
