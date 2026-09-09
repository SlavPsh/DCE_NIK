"""FAIR redo: conventional CS Patlak-fit at f25 (same data as NIK-PK), not f100. recompute
inter-slice CoV and a spatial-noise metric, and a side-by-side visual, all at f25."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, os, sys, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, ".")
from figpath import fig as fpath
import consolidated as C
D = "/net/beegfs/users/P101440/DCE_NIK"; CSD = "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs"; TA = 375.0
SLICES = [18, 19, 21]; ROIS = ["aorta", "cortex", "medulla", "liver"]

def basis_on(Z, nt):
    az = np.load(f"{D}/aif_slice{Z}.npz"); tg = np.linspace(0, TA, nt)
    aif = np.interp(tg, np.asarray(az["tC"]), np.asarray(az["aif_frame"])); aif = aif/(aif.max()+1e-9)
    iaif = np.concatenate([[0], np.cumsum(0.5*(aif[1:]+aif[:-1])*np.diff(tg))]); iaif/=(iaif.max()+1e-9)
    return np.linalg.pinv(np.stack([aif, iaif, np.ones_like(aif)], 1))
def maps(mag, Bpinv): amp = np.tensordot(mag, Bpinv.T, axes=([2],[0])); return np.abs(amp[...,0]), np.abs(amp[...,1])

res = {}; sp_noise = {}
for Z in SLICES:
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; BODY = ctx["BODY"]; AIR = ctx["AIR"]
    cs25 = np.abs(np.load(f"{CSD}/cs_slice{Z:02d}_f25.npy")).astype(np.float32)      # CS at f25 (fair)
    vp_cs, kt_cs = maps(cs25, basis_on(Z, cs25.shape[-1]))
    nik = np.abs(np.load(f"{D}/results_batch/pk_f0_sl{Z}/nik_slice_{Z}_cplx.npy")).astype(np.float32)
    vp_n, kt_n = maps(nik, basis_on(Z, nik.shape[-1]))
    res[Z] = dict(vp_cs=vp_cs, kt_cs=kt_cs, vp_n=vp_n, kt_n=kt_n,
                  roi_cs={r: dict(vp=float(vp_cs[rois[r]].mean()), kt=float(kt_cs[rois[r]].mean())) for r in ROIS if rois[r].sum()>0},
                  roi_n={r: dict(vp=float(vp_n[rois[r]].mean()), kt=float(kt_n[rois[r]].mean())) for r in ROIS if rois[r].sum()>0})
    # spatial-noise metric: air-region std / body-mean of the Ktrans map (higher = noisier map)
    sp_noise[Z] = dict(NIK=float(kt_n[AIR].std()/(kt_n[BODY].mean()+1e-12)), CS_f25=float(kt_cs[AIR].std()/(kt_cs[BODY].mean()+1e-12)))

def cov(v): v=np.array(v); return float(np.std(v)/(np.mean(v)+1e-12)*100)
print("=== FAIR inter-slice CoV%% (both f25): Ktrans / vp, NIK-PK F0 vs CS-fit(f25) ===")
print(f"{'tissue':>8} {'Kt_NIK':>7} {'Kt_CSf25':>9} {'vp_NIK':>7} {'vp_CSf25':>9}")
for tis in ROIS:
    kN=cov([res[Z]['roi_n'][tis]['kt'] for Z in SLICES]); kC=cov([res[Z]['roi_cs'][tis]['kt'] for Z in SLICES])
    vN=cov([res[Z]['roi_n'][tis]['vp'] for Z in SLICES]); vC=cov([res[Z]['roi_cs'][tis]['vp'] for Z in SLICES])
    print(f"{tis:>8} {kN:>7.0f} {kC:>9.0f} {vN:>7.0f} {vC:>9.0f}")
print("\n=== spatial-noise of Ktrans map (air-std / body-mean; higher=noisier) ===")
for Z in SLICES: print(f"  sl{Z}: NIK {sp_noise[Z]['NIK']:.3f}   CS_f25 {sp_noise[Z]['CS_f25']:.3f}")

# fair visual: NIK vs CS both f25
fig, ax = plt.subplots(2, len(SLICES), figsize=(4*len(SLICES), 8))
for j, Z in enumerate(SLICES):
    kn=res[Z]['kt_n']; kc=res[Z]['kt_cs']; vmax=np.percentile(kc,99)
    ax[0,j].imshow(np.rot90(kn), cmap="viridis", vmax=vmax); ax[0,j].set_title(f"sl{Z} NIK-PK F0 Ktrans (f25)", fontsize=9); ax[0,j].axis("off")
    ax[1,j].imshow(np.rot90(kc), cmap="viridis", vmax=vmax); ax[1,j].set_title(f"sl{Z} CS-fit Ktrans (f25, FAIR)", fontsize=9); ax[1,j].axis("off")
fig.suptitle("FAIR comparison: NIK-PK vs CS-fit, BOTH at f25 (same data)", fontweight="bold")
fig.tight_layout(); p=fpath("task2_fair_f25.png"); fig.savefig(p, dpi=130); print(f"\nwrote {p.split('/')[-1]}")
