"""is the 'anatomy gone / noisier' in NIK-PK a Patlak-F0 rigidity issue or NIK itself?
compare Ktrans maps (f25, sl18/19/21): NIK-PK F0, NIK-PK F2, NIK-full projected onto Patlak,
and CS-fit(f25). quantify structure INSIDE the body (masked SSIM vs CS-fit as a smoother
reference, + body-interior gradient/detail energy). out: figure + table."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, sys, torch, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, ".")
from figpath import fig as fpath
import consolidated as C
from masked_metrics import ssim_masked
D = "/net/beegfs/users/P101440/DCE_NIK"; CSD = "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs"; TA = 375.0; dev = "cpu"
SLICES = [18, 19, 21]

def bpinv(Z, nt):
    az = np.load(f"{D}/aif_slice{Z}.npz"); tg = np.linspace(0, TA, nt)
    aif = np.interp(tg, np.asarray(az["tC"]), np.asarray(az["aif_frame"])); aif /= (aif.max()+1e-9)
    iaif = np.concatenate([[0], np.cumsum(0.5*(aif[1:]+aif[:-1])*np.diff(tg))]); iaif /= (iaif.max()+1e-9)
    return np.linalg.pinv(np.stack([aif, iaif, np.ones_like(aif)], 1))
def ktmap(mag, Z): amp = np.tensordot(mag, bpinv(Z, mag.shape[-1]).T, axes=([2],[0])); return np.abs(amp[...,1])
def load(p): return np.abs(np.load(p)).astype(np.float32)

def masked_ssim(a, b, m):
    va = np.percentile(b[m], 99.5)
    ta = torch.from_numpy(np.clip(a/(va+1e-9),0,1)[None,None]).float(); tb = torch.from_numpy(np.clip(b/(va+1e-9),0,1)[None,None]).float()
    mt = torch.from_numpy(m.astype(np.float32))[None,None]
    return float(ssim_masked(ta, tb, mt, data_range=1.0))

rows = {}; MAPS = {}
for Z in SLICES:
    ctx = C.slice_ctx(Z); BODY = ctx["BODY"]
    m = {}
    m["CS_f25"] = ktmap(load(f"{CSD}/cs_slice{Z:02d}_f25.npy"), Z)
    m["NIK_PK_F0"] = ktmap(load(f"{D}/results_batch/pk_f0_sl{Z}/nik_slice_{Z}_cplx.npy"), Z)
    m["NIK_PK_F2"] = ktmap(load(f"{D}/results_batch/pk_f2_sl{Z}/nik_slice_{Z}_cplx.npy"), Z)
    m["NIK_full_proj"] = ktmap(load(f"{D}/results_batch/full_sl{Z}{'f25' if Z==21 else ''}/nik_slice_{Z}_cplx.npy"), Z)
    MAPS[Z] = (m, BODY)
    # interior detail energy (grad magnitude inside eroded body) and SSIM vs CS_f25 reference
    inner = ndi.binary_erosion(BODY, iterations=4)
    for k, v in m.items():
        g = ndi.gaussian_gradient_magnitude(v, 1.0)
        rows.setdefault(k, {})[Z] = dict(detail=float(g[inner].mean()/(v[BODY].mean()+1e-12)),
                                         ssim_vs_CS=masked_ssim(v, m["CS_f25"], BODY))

print("=== structure INSIDE body: interior detail (grad/mean; higher=more structure OR noise) + SSIM vs CS_f25 ===")
print(f"{'map':>14}" + "".join(f"  sl{Z}_detail sl{Z}_ssim" for Z in SLICES))
for k in ["CS_f25", "NIK_full_proj", "NIK_PK_F2", "NIK_PK_F0"]:
    print(f"{k:>14}" + "".join(f"  {rows[k][Z]['detail']:>9.2f} {rows[k][Z]['ssim_vs_CS']:>8.2f}" for Z in SLICES))

# figure: 4 maps x 3 slices
fig, ax = plt.subplots(4, len(SLICES), figsize=(4*len(SLICES), 14)); order = ["CS_f25", "NIK_full_proj", "NIK_PK_F2", "NIK_PK_F0"]
for i, k in enumerate(order):
    for j, Z in enumerate(SLICES):
        m, BODY = MAPS[Z]; v = m[k]; vmax = np.percentile(m["CS_f25"], 99)
        ax[i, j].imshow(np.rot90(v), cmap="viridis", vmax=vmax); ax[i, j].axis("off")
        ax[i, j].set_title(f"sl{Z} {k}" + (f"  ssim{rows[k][Z]['ssim_vs_CS']:.2f}" if k!="CS_f25" else ""), fontsize=8.5)
fig.suptitle("Ktrans maps, all f25: does anatomy return with more DoF? (F0 rigid Patlak -> F2 -> full-rank -> CS-fit)", fontweight="bold")
fig.tight_layout(); p = fpath("task2_anatomy_f25.png"); fig.savefig(p, dpi=130); print(f"\nwrote {p.split('/')[-1]}")
