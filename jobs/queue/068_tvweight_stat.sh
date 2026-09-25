#!/bin/bash
#SBATCH -J tvwstat
#SBATCH -p defq
#SBATCH -c 8
#SBATCH --mem 32G
#SBATCH -t 0:40:00
# precompute_ref.recon_slice sets TVWeight1/2 = |E.H y|.max() * Weight. what does .mean() give instead?
# measures max, mean, median and percentiles of the subspace adjoint per slice, the mean/max ratio, and how much of
# the grid is background (which is what drags the mean down). no recon is changed; read-only diagnostic.
set -uo pipefail
P=/net/beegfs/users/P101440/grasp_pro_py
export MAMBA_ROOT_PREFIX=/net/beegfs/users/P101440/micromamba PATH=/net/beegfs/users/P101440/micromamba/bin:$PATH
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
echo "$(date '+%F %T') host $(hostname) job ${SLURM_JOB_ID:-?}"
cd $P && git log --oneline -1
micromamba run -n torch29 python -u - <<'PY'
import sys, numpy as np
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
import precompute_ref as pr
from dce.recon import build_phi
R = "/net/beegfs/users/P101440/dce_data/orig/gv2_DCE_Rerun/prep"
sh = np.load(f"{R}/shared.npz"); T = np.asarray(sh["traj_grog"]); bas = int(sh["bas"])
NLINE, K = 21, 5
nt = T.shape[1] // NLINE
print(f"{nt} frames x {NLINE} spokes, K={K}, Weight1={pr.Weight1}, Weight2={pr.Weight2}")
print(f"{'slice':>5} {'max':>11} {'mean':>11} {'median':>11} {'p99':>11} {'mean/max':>9} {'nonzero%':>9}")
rat = []
for z in (13, 27, 40):
    kdr = np.load(f"{R}/slice_{z:02d}.npz")["kdata_radial"]
    Phi = build_phi(kdr, nt, NLINE, K)
    # front half of recon_slice, verbatim, up to the adjoint
    nx, ntv, nc = kdr.shape
    D = kdr.reshape(nx * ntv, nc, order="F"); Vcc = np.linalg.svd(D, full_matrices=False)[2].conj().T[:, :pr.ncc]
    kdc = (D @ Vcc).reshape(nx, ntv, pr.ncc, order="F").astype(np.complex64); kk = kdc[:, :, None, :]
    Gx, Gy = pr.get_gx_gy(kk, T)
    kref, _ = pr.grog_dictionary_interp(kk, Gx, Gy, T, 1)
    b1 = pr.adapt_array_2d(np.squeeze(pr.ifft2c_mri(kref))); b1 = (b1 / np.abs(b1).max()).astype(np.complex64)
    _, DCF = pr.grog_dictionary_interp(kk[:, -pr.Nqu:], Gx, Gy, T[:, -pr.Nqu:], 0)
    kdata2 = kdc[:, :nt*NLINE, :].reshape(nx, NLINE, nt, pr.ncc, order="F")
    Traj2 = T[:, :nt*NLINE].reshape(nx, NLINE, nt, order="F")
    kdata3, DCF_U = pr.grog_dictionary_interp(kdata2, Gx, Gy, Traj2, 1)
    mask = (kdata3[:, :, :, 0] != 0).astype(np.complex64)
    Weightc = np.repeat(DCF, nt, axis=2) / DCF_U
    PCA = pr.TempPCASub(Phi)
    kdatac = np.stack([PCA @ kdata3[:, :, :, i] for i in range(pr.ncc)], axis=3)
    Wc = PCA @ Weightc; Wc = np.repeat(Wc[:, :, :1], K, axis=2)[:, :, :, None]; Wc = np.repeat(Wc, pr.ncc, axis=3)
    E = pr.Emat_GROG2Dksp(mask, b1, Wc, PCA, 1)
    y = (kdatac * np.sqrt(Wc)).astype(np.complex64)
    rec = np.abs(E.H @ y)
    r = rec.mean() / rec.max(); rat.append(r)
    print(f"{z:5d} {rec.max():11.4e} {rec.mean():11.4e} {np.median(rec):11.4e} {np.percentile(rec,99):11.4e} "
          f"{r:9.4f} {100*(rec > 0.01*rec.max()).mean():9.1f}")
    print(f"        -> TVWeight1 with max {rec.max()*pr.Weight1:.4e}   with mean {rec.mean()*pr.Weight1:.4e}"
          f"   (weaker by {1/r:.0f}x)")
    print(f"        -> per coefficient map max: {[f'{rec[:,:,k].max():.2e}' for k in range(K)]}", flush=True)
print(f"\nmean/max over slices: {np.mean(rat):.4f} -> switching max to mean divides both TV weights by ~{1/np.mean(rat):.0f}")
print("TVWEIGHT_STAT_DONE")
PY
