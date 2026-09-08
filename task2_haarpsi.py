"""same Ktrans-map comparison as task2_wholepicture but with MORE metrics: body-masked HaarPSI
(perceptual, less grain/streak-biased than SSIM), scale-invariant Pearson correlation, and
LS-scaled PSNR. references: each method's own f100 AND the CS-f100 gold standard. reuses
masked_metrics.haarpsi_masked / ssim_masked (no new metric code)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys, os, torch
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); sys.path.insert(0, ".")
import consolidated as C
from masked_metrics import haarpsi_masked, ssim_masked
D = "/scratch/rnga/vvpshenov/DCE_NIK"; CSD = "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs"; TA = 375.0; SLICES = [18, 19, 21]
def bpinv(Z, nt):
    az = np.load(f"{D}/aif_slice{Z}.npz"); tg = np.linspace(0, TA, nt)
    aif = np.interp(tg, np.asarray(az["tC"]), np.asarray(az["aif_frame"])); aif /= (aif.max()+1e-9)
    iaif = np.concatenate([[0], np.cumsum(0.5*(aif[1:]+aif[:-1])*np.diff(tg))]); iaif /= (iaif.max()+1e-9)
    return np.linalg.pinv(np.stack([aif, iaif, np.ones_like(aif)], 1))
def kt(mag, Z): return np.abs(np.tensordot(mag, bpinv(Z, mag.shape[-1]).T, axes=([2],[0]))[..., 1])
def L(p): return np.abs(np.load(p)).astype(np.float32) if os.path.exists(p) else None
def tb(a, v): return torch.from_numpy(np.clip(a/(v+1e-9),0,1)[None,None]).float()
def metrics(a, ref, body):
    v = np.percentile(ref[body], 99.5); mt = torch.from_numpy(body.astype(np.float32))[None,None]
    h = float(haarpsi_masked(tb(a, v), tb(ref, v), mt, data_range=1.0)); s = float(ssim_masked(tb(a, v), tb(ref, v), mt, data_range=1.0))
    corr = float(np.corrcoef(a[body], ref[body])[0, 1])
    sc = float((a[body] @ ref[body]) / (a[body] @ a[body] + 1e-12)); mse = float(((a[body]*sc - ref[body])**2).mean())
    psnr = 10*np.log10(float(ref[body].max())**2/(mse+1e-20))
    return h, s, corr, psnr

rows = {}
for Z in SLICES:
    ctx = C.slice_ctx(Z); body = ctx["BODY"]
    maps = {"NIK_F0_f25": L(f"{D}/results_batch/pk_f0_sl{Z}/nik_slice_{Z}_cplx.npy"),
            "NIK_F2_f25": L(f"{D}/results_batch/pk_f2_sl{Z}/nik_slice_{Z}_cplx.npy"),
            "CS_f25": L(f"{CSD}/cs_slice{Z:02d}_f25.npy")}
    csf100 = ctx["cs100"]; ref = kt(csf100, Z)
    for k, m in maps.items():
        if m is None: continue
        rows.setdefault(k, {})[Z] = metrics(kt(m, Z), ref, body)

print("=== Ktrans map vs CS-f100 (gold standard), body-masked. HaarPSI | SSIM | corr | PSNR(dB) ===")
print(f"{'method':>12}" + "".join(f"{'sl'+str(Z):>26}" for Z in SLICES))
for k in ["NIK_F0_f25", "NIK_F2_f25", "CS_f25"]:
    cells = "".join(f"  H{rows[k][Z][0]:.2f} S{rows[k][Z][1]:.2f} r{rows[k][Z][2]:.2f} P{rows[k][Z][3]:4.1f}" for Z in SLICES if Z in rows.get(k, {}))
    print(f"{k:>12}{cells}")
print("\n=== means across slices ===")
print(f"{'method':>12}{'HaarPSI':>9}{'SSIM':>7}{'corr':>7}{'PSNR':>7}")
for k in ["NIK_F0_f25", "NIK_F2_f25", "CS_f25"]:
    arr = np.array([rows[k][Z] for Z in SLICES if Z in rows.get(k, {})]).mean(0)
    print(f"{k:>12}{arr[0]:>9.2f}{arr[1]:>7.2f}{arr[2]:>7.2f}{arr[3]:>7.1f}")
