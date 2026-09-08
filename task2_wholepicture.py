"""whole picture: Ktrans-map quality vs a REAL reference (each method's own f100).
compares NIK-PK f25 vs NIK-PK f100 (degradation under acceleration), NIK-PK f25 vs CS f100
(accelerated NIK vs gold-standard CS), and CS f25 vs CS f100 (CS's own degradation). answers:
does F0 grain VANISH at f100 (undersampling noise) or PERSIST (Patlak rigidity)?
out: table + figure (F2 primary). sl18/19/21."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys, os, torch, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); sys.path.insert(0, ".")
from figpath import fig as fpath
import consolidated as C
from masked_metrics import ssim_masked
D = "/scratch/rnga/vvpshenov/DCE_NIK"; REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
CSD = "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs"; TA = 375.0; SLICES = [18, 19, 21]

def bpinv(Z, nt):
    az = np.load(f"{D}/aif_slice{Z}.npz"); tg = np.linspace(0, TA, nt)
    aif = np.interp(tg, np.asarray(az["tC"]), np.asarray(az["aif_frame"])); aif /= (aif.max()+1e-9)
    iaif = np.concatenate([[0], np.cumsum(0.5*(aif[1:]+aif[:-1])*np.diff(tg))]); iaif /= (iaif.max()+1e-9)
    return np.linalg.pinv(np.stack([aif, iaif, np.ones_like(aif)], 1))
def kt(mag, Z): return np.abs(np.tensordot(mag, bpinv(Z, mag.shape[-1]).T, axes=([2],[0]))[..., 1])
def L(p): return np.abs(np.load(p)).astype(np.float32) if os.path.exists(p) else None
def mssim(a, b, m):
    v = np.percentile(b[m], 99.5)
    ta = torch.from_numpy(np.clip(a/(v+1e-9),0,1)[None,None]).float(); tb = torch.from_numpy(np.clip(b/(v+1e-9),0,1)[None,None]).float()
    return float(ssim_masked(ta, tb, torch.from_numpy(m.astype(np.float32))[None,None], data_range=1.0))
def grain(v, inner, body): g = ndi.gaussian_gradient_magnitude(v, 1.0); return float(g[inner].mean()/(v[body].mean()+1e-12))

for F in [0, 2]:
    print(f"\n===== NIK-PK F{F}: Ktrans-map quality vs REAL references (SSIM), interior grain =====")
    print(f"{'slice':>5} {'NIKf25-vs-NIKf100':>18} {'NIKf25-vs-CSf100':>17} {'CSf25-vs-CSf100':>16} | {'grain NIKf25/NIKf100/CSf25/CSf100':>34}")
    for Z in SLICES:
        ctx = C.slice_ctx(Z); body = ctx["BODY"]; inner = ndi.binary_erosion(body, iterations=4)
        nik25 = L(f"{D}/results_batch/pk_f{F}_sl{Z}/nik_slice_{Z}_cplx.npy")
        nik100 = L(f"{D}/results_batch/pk_f{F}_sl{Z}_f100/nik_slice_{Z}_cplx.npy")
        cs25 = L(f"{CSD}/cs_slice{Z:02d}_f25.npy"); cs100 = ctx["cs100"]
        if nik25 is None or nik100 is None: print(f"{Z:>5}  MISSING f100"); continue
        kN25, kN100, kC25, kC100 = kt(nik25, Z), kt(nik100, Z), kt(cs25, Z), kt(cs100, Z)
        s_nn = mssim(kN25, kN100, body); s_nc = mssim(kN25, kC100, body); s_cc = mssim(kC25, kC100, body)
        gr = [grain(x, inner, body) for x in (kN25, kN100, kC25, kC100)]
        print(f"{Z:>5} {s_nn:>18.2f} {s_nc:>17.2f} {s_cc:>16.2f} | {gr[0]:>8.2f}{gr[1]:>8.2f}{gr[2]:>8.2f}{gr[3]:>8.2f}")

# figure: F2, per slice: NIK f25 | NIK f100 | CS f25 | CS f100
F = 2; fig, ax = plt.subplots(len(SLICES), 4, figsize=(16, 4*len(SLICES))); cols = ["NIK-PK f25", "NIK-PK f100", "CS f25", "CS f100"]
for i, Z in enumerate(SLICES):
    ctx = C.slice_ctx(Z)
    mats = [L(f"{D}/results_batch/pk_f{F}_sl{Z}/nik_slice_{Z}_cplx.npy"), L(f"{D}/results_batch/pk_f{F}_sl{Z}_f100/nik_slice_{Z}_cplx.npy"),
            L(f"{CSD}/cs_slice{Z:02d}_f25.npy"), ctx["cs100"]]
    kts = [kt(m, Z) if m is not None else None for m in mats]; vmax = np.percentile(kts[3], 99)
    for j, (c, k) in enumerate(zip(cols, kts)):
        if k is None: ax[i, j].axis("off"); continue
        ax[i, j].imshow(np.rot90(k), cmap="viridis", vmax=vmax); ax[i, j].axis("off")
        if i == 0: ax[i, j].set_title(c, fontsize=11)
        ax[i, j].text(3, 15, f"sl{Z}", color="w", fontsize=8)
fig.suptitle("Ktrans (F2): NIK-PK f25 -> f100 vs CS f25 -> f100. f100 = full-data reference for each method", fontweight="bold")
fig.tight_layout(); p = fpath("task2_wholepicture_f25_f100.png"); fig.savefig(p, dpi=125); print(f"\nwrote {p.split('/')[-1]}")
