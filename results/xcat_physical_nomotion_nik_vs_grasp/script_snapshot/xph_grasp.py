"""GRASP-Pro (faithful algorithm) on the physical no-motion XCAT: k-centre-navigator PCA temporal
subspace (K=5) + spatial-TV + temporal-TV CS, on the SAME train spokes / coils / trajectory as NIK.
Operator = cufinufft SENSE-NUFFT (the established GROG gridding is replaced by NUFFT; documented).
Weights match the real GRASP-Pro: spatial 0.0005, temporal 0.001; ~15 CG-equivalent iters."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, cupy as cp
import task4_gpu_recon as G          # GPUSub (cufinufft SENSE-NUFFT subspace) + recon (FISTA SP/T Huber-TV)
import xph_pipeline as P
K_PCA = 5; SPATIAL_TV = 0.0005; TEMPORAL_TV = 0.001; ITERS = 15

def build_phi(K=K_PCA):
    """K temporal PCA components from the k-space-centre navigator (data-driven; GRASP-Pro convention)."""
    d = P.data(); RO = d["kdata"].shape[-1]; c0 = RO // 2
    nav = np.abs(d["kdata"][:, :, :, c0 - 2:c0 + 3])          # (C,F,nang,5)
    F = nav.shape[1]; ds = nav.transpose(0, 2, 3, 1).reshape(-1, F)   # (obs, F)
    w, PC = np.linalg.eigh(np.cov(ds, rowvar=False))          # (F,F) temporal cov
    Phi = PC[:, np.argsort(-w)][:, :K].astype(np.complex64)   # (F,K)
    return Phi

def reconstruct(iters=ITERS, ls=SPATIAL_TV, lt=TEMPORAL_TV):
    """GRASP-Pro dynamic (complex, coil-combined) at the native frame times -> [RO,RO,F] + Phi."""
    d = P.data(); b1 = d["b1"]; kx = d["kx"]; ky = d["ky"]; F, nang, RO = kx.shape
    Phi = build_phi()
    tr = P.TRAIN_ANG
    trajs = [(-kx[t, tr].ravel(), -ky[t, tr].ravel()) for t in range(F)]   # negate -> +2pi convention (validated)
    y = [d["kdata"][:, t, tr, :].reshape(b1.shape[-1], -1).astype(np.complex64) for t in range(F)]
    E = G.GPUSub(Phi, b1, trajs)
    coeff = G.recon(E, y, iters=iters, ls=ls, lt=lt, label="grasp")        # [RO,RO,K]
    dyn = np.einsum("xyk,tk->xyt", coeff, Phi)                             # [RO,RO,F] complex
    return dyn.astype(np.complex64), Phi, coeff

if __name__ == "__main__":
    import xph_common as X
    dyn, Phi, _ = reconstruct(iters=6)   # quick smoke
    tq = P.data()["times"]; T = X.truth_at(P.ZI, tq); body = P.data()["labels"] > 0
    rm = np.abs(dyn).mean(2); tm = T.mean(2)
    for nm, fn in [("identity", lambda x: x), ("rot180", lambda x: x[::-1, ::-1]), ("fliplr", np.fliplr), ("flipud", np.flipud)]:
        print("  grasp orient %-8s corr %.3f" % (nm, np.corrcoef(fn(rm)[body].ravel(), tm[body].ravel())[0, 1]))
    print("Phi shape", Phi.shape, "| grasp dyn", dyn.shape)
