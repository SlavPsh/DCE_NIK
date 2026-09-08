"""GRASP-Pro on the XCAT phantom: temporal PCA subspace (library TempPCASub) + library NLCG solver
(cs_l1_nlcg_sptv) + spatial/temporal TV, driven by a NUFFT-SENSE subspace operator using the TRUE coils
from the file + ramp dcf. GROG replaced (needs coil phase the phantom lacks); rest is grasp_pro_py.
operator is power-iteration-normalized so ||E.H E||~1 (stable NLCG)."""
import warnings; warnings.filterwarnings("ignore")
import sys, copy, numpy as np, cupy as cp, cufinufft
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import precompute_ref as pr
import xph_pipeline as P, xph_common as X
K = 5; NITE = 5; NOUTER = 3; SIGN = -1.0; EPS = 1e-5

def build_phi(kdata, tr, K=K):
    C, F, nang, RO = kdata.shape; c0 = RO // 2
    nav = np.abs(kdata[:, :, tr, c0 - 2:c0 + 3]); ds = nav.transpose(0, 2, 3, 1).reshape(-1, F)
    w, PC = np.linalg.eigh(np.cov(ds, rowvar=False))
    return PC[:, np.argsort(-w)][:, :K].astype(np.complex64)

class Emat_NUFFT:
    adjoint = False
    def __init__(self, trajs, dcf, b1, Phi, N):
        self.N = N; self.C = b1.shape[-1]; self.nt = len(trajs); self.K = Phi.shape[1]
        b1c = cp.asarray(b1, cp.complex128); nrm = cp.sqrt(cp.max(cp.sum(cp.abs(b1c) ** 2, -1)))
        self.b1 = b1c / (nrm + 1e-12); dsum = cp.sum(cp.abs(self.b1) ** 2, -1)
        self.den = dsum + 1e-3 * float(dsum.max())
        self.Phi = cp.asarray(Phi, cp.complex128)
        self.co = [(cp.asarray(SIGN * 2 * np.pi * kx.ravel(), cp.float64), cp.asarray(SIGN * 2 * np.pi * ky.ravel(), cp.float64)) for (kx, ky) in trajs]
        self.sdcf = [cp.asarray(np.sqrt(w).ravel(), cp.float64) for w in dcf]
        self.p2 = cufinufft.Plan(2, (N, N), self.C, eps=EPS, isign=-1, dtype="complex128")
        self.p1 = cufinufft.Plan(1, (N, N), self.C, eps=EPS, isign=1, dtype="complex128")
        v = cp.asarray(np.random.randn(N, N, self.K) + 1j * np.random.randn(N, N, self.K))     # normalize ||E.H E||~1
        for _ in range(5):
            w = self._adj_cp(self._fwd_cp(v)); v = w / (cp.linalg.norm(w) + 1e-12)
        lam = float(cp.real(cp.vdot(v, self._adj_cp(self._fwd_cp(v)))) + 1e-12)
        self.a = 1.0 / np.sqrt(lam); self.sdcf = [s * self.a for s in self.sdcf]
    @property
    def H(self):
        o = copy.copy(self); o.adjoint = not self.adjoint; return o
    def __matmul__(self, b):
        return cp.asnumpy(self._adj_cp(cp.asarray(b, cp.complex128))) if self.adjoint else cp.asnumpy(self._fwd_cp(cp.asarray(b, cp.complex128)))
    def _fwd_cp(self, zc):
        out = []
        for t in range(self.nt):
            img = (zc * self.Phi[t][None, None, :]).sum(-1)
            cimg = cp.ascontiguousarray(cp.transpose(img[:, :, None] * self.b1, (2, 0, 1)))
            self.p2.setpts(self.co[t][0], self.co[t][1])
            out.append(self.p2.execute(cimg) * self.sdcf[t][None, :])
        return cp.stack(out, 1)
    def _adj_cp(self, yc):
        z = cp.zeros((self.N, self.N, self.K), cp.complex128)
        for t in range(self.nt):
            self.p1.setpts(self.co[t][0], self.co[t][1])
            cimg = self.p1.execute(cp.ascontiguousarray(yc[:, t, :] * self.sdcf[t][None, :]))
            img = (cp.conj(self.b1) * cp.transpose(cimg, (1, 2, 0))).sum(-1) / self.den
            for k in range(self.K): z[:, :, k] += cp.conj(self.Phi[t, k]) * img
        return z
    def apply_dcf(self, raw):                                                                  # raw [C,nt,M] -> weighted y (same sdcf as operator)
        return np.stack([raw[:, t, :] * cp.asnumpy(self.sdcf[t])[None, :] for t in range(self.nt)], 1)

def reconstruct(zi=None, k=K):
    zi = P.ZI if zi is None else zi
    d = X.load_slice(zi); kx = d["kx"]; ky = d["ky"]; kdata = d["kdata"]; b1 = d["b1"]
    C, F, nang, RO = kdata.shape; tr = np.array(P.TRAIN_ANG)
    trajs = [(-kx[t, tr], -ky[t, tr]) for t in range(F)]
    dcf = [np.maximum(np.abs(kx[t, tr] + 1j * ky[t, tr]), 1e-3) for t in range(F)]
    Phi = build_phi(kdata, tr, K=k); PCA = pr.TempPCASub(Phi)
    E = Emat_NUFFT(trajs, dcf, b1, Phi, RO)
    raw = np.stack([kdata[:, t, tr, :].reshape(C, -1) for t in range(F)], 1).astype(np.complex128)
    y = E.apply_dcf(raw)
    recon = E.H @ y
    print("op-scale a=%.3e | init recon finite=%s |recon|max %.3e |y|max %.3e" % (E.a, np.isfinite(recon).all(), np.abs(recon).max(), np.abs(y).max()), flush=True)
    param = dict(E=E, y=y, PCA=PCA, TV1=pr.TV_Temp(), TV2=pr.FD1OP(),
                 TVWeight1=np.abs(recon).max() * pr.Weight1, TVWeight2=np.abs(recon).max() * pr.Weight2, nite=NITE)
    for it in range(NOUTER):
        recon = pr.cs_l1_nlcg_sptv(recon, param); print("  nlcg outer %d/%d finite=%s" % (it + 1, NOUTER, np.isfinite(recon).all()), flush=True)
    dyn = np.asarray(PCA.H @ recon)
    return dyn.astype(np.complex64), Phi
