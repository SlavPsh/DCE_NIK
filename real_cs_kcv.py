"""J1a: held-out k-space cross-validation for the temporal-PCA rank K on the REAL in-vivo data
(slice 13), same protocol as the phantom. The real-data CS reference uses K=5 (variance default). Recon
at each K from a TRAIN subset of spokes (10/14 per frame), forward-project to the MEASURED-but-held-out
spokes (4/14), pick K* = argmin held-out NMSE. NUFFT-SENSE recon (TRUE coils + ramp dcf + TempPCASub +
grasp_pro_py NLCG solver), CPU finufft, coils batched. Ground truth NOT used (none exists in vivo)."""
import warnings; warnings.filterwarnings("ignore")
import sys, copy, numpy as np, finufft
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
import precompute_ref as pr
KS = [3, 5, 8, 12, 16, 24, 32]; NLINE = 14; NITE = 4; NOUTER = 2; EPS = 1e-5
REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"
sh = np.load(f"{REF}/shared.npz"); s13 = np.load(f"{REF}/slice_13.npz")
tn = np.asarray(sh["traj_norm"]); b1 = np.asarray(s13["b1"]).astype(np.complex64)
kdr = np.asarray(s13["kdata_radial"]).astype(np.complex64)                                  # [nx,1710,ncc]
nx, nv, ncc = kdr.shape; N = b1.shape[0]; NT = nv // NLINE; c0 = nx // 2
kx_all = (2 * np.pi * tn.real).astype(np.float64); ky_all = (2 * np.pi * tn.imag).astype(np.float64)
vidx = np.arange(NT * NLINE).reshape(NLINE, NT, order="F"); TRAIN = vidx[:10]; HOLD = vidx[10:]
b1n = (b1 / (np.sqrt(np.max(np.sum(np.abs(b1) ** 2, -1))) + 1e-12)).astype(np.complex128)    # [N,N,C]
b1c = np.ascontiguousarray(np.transpose(b1n, (2, 0, 1)))                                     # [C,N,N]
print(f"real slice13: nx {nx} N {N} ncc {ncc} | NT {NT} frames x {NLINE} | train 10 / holdout 4 per frame", flush=True)

def freqs(cols, t): return np.ascontiguousarray(kx_all[:, cols[:, t]].reshape(-1)), np.ascontiguousarray(ky_all[:, cols[:, t]].reshape(-1))
def meas(cols, t): return np.ascontiguousarray(kdr[:, cols[:, t], :].transpose(2, 0, 1).reshape(ncc, -1))   # [C, nx*nspoke]
def build_phi(K, cols):
    nav = np.abs(kdr[c0-2:c0+3][:, cols.reshape(-1, order="F"), :]).reshape(5, cols.shape[0], NT, ncc, order="F").mean(1)
    ds = nav.transpose(0, 2, 1).reshape(5 * ncc, NT, order="F")
    w, PC = np.linalg.eigh(np.cov(ds, rowvar=False)); return PC[:, np.argsort(-w)][:, :K].astype(np.complex64)

class Emat:
    adjoint = False
    def __init__(self, cols, Phi):
        self.N = N; self.C = ncc; self.nt = NT; self.K = Phi.shape[1]
        dsum = np.sum(np.abs(b1n) ** 2, -1); self.den = dsum + 1e-3 * float(dsum.max()); self.Phi = Phi.astype(np.complex128)
        self.fxy = [freqs(cols, t) for t in range(NT)]
        self.sdcf = [np.sqrt(np.maximum(np.abs(kx_all[:, cols[:, t]] + 1j*ky_all[:, cols[:, t]]).reshape(-1)/(2*np.pi), 1e-3)) for t in range(NT)]
        v = np.random.randn(N, N, self.K) + 1j*np.random.randn(N, N, self.K)
        for _ in range(4): w = self._adj(self._fwd(v)); v = w/(np.linalg.norm(w)+1e-12)
        lam = float(np.real(np.vdot(v, self._adj(self._fwd(v))))+1e-12); self.a = 1.0/np.sqrt(lam); self.sdcf = [s*self.a for s in self.sdcf]
    @property
    def H(self): o = copy.copy(self); o.adjoint = not self.adjoint; return o
    def __matmul__(self, b): return self._adj(b) if self.adjoint else self._fwd(b)
    def _fwd(self, z):
        out = np.empty((self.C, self.nt, self.fxy[0][0].size), np.complex128)
        for t in range(self.nt):
            img = (z * self.Phi[t][None, None, :]).sum(-1)
            cimg = np.ascontiguousarray(img[None] * b1c)                                     # [C,N,N]
            out[:, t] = finufft.nufft2d2(self.fxy[t][0], self.fxy[t][1], cimg, isign=-1, eps=EPS) * self.sdcf[t][None, :]
        return out
    def _adj(self, y):
        z = np.zeros((self.N, self.N, self.K), np.complex128)
        for t in range(self.nt):
            cin = np.ascontiguousarray(y[:, t] * self.sdcf[t][None, :])                      # [C,M]
            cimg = finufft.nufft2d1(self.fxy[t][0], self.fxy[t][1], cin, (self.N, self.N), isign=1, eps=EPS)  # [C,N,N]
            img = (np.conj(b1n) * np.transpose(cimg, (1, 2, 0))).sum(-1) / self.den
            for k in range(self.K): z[:, :, k] += np.conj(self.Phi[t, k]) * img
        return z
    def apply_dcf(self, raw): return np.stack([raw[:, t] * self.sdcf[t][None, :] for t in range(self.nt)], 1)

rows = []
for K in KS:
    Phi = build_phi(K, TRAIN); PCA = pr.TempPCASub(Phi); E = Emat(TRAIN, Phi)
    raw = np.stack([meas(TRAIN, t) for t in range(NT)], 1).astype(np.complex128)             # [C,nt,M]
    y = E.apply_dcf(raw); recon = E.H @ y
    param = dict(E=E, y=y, PCA=PCA, TV1=pr.TV_Temp(), TV2=pr.FD1OP(), TVWeight1=np.abs(recon).max()*pr.Weight1, TVWeight2=np.abs(recon).max()*pr.Weight2, nite=NITE)
    for _ in range(NOUTER): recon = pr.cs_l1_nlcg_sptv(recon, param)
    dyn = np.asarray(PCA.H @ recon)                                                          # [N,N,NT]
    preds, meass, rads = [], [], []
    for t in range(NT):
        fx, fy = freqs(HOLD, t); cimg = np.ascontiguousarray(dyn[:, :, t][None] * b1c)
        preds.append(finufft.nufft2d2(fx, fy, cimg, isign=-1, eps=EPS)); meass.append(meas(HOLD, t))
        rr = np.abs(kx_all[:, HOLD[:, t]] + 1j*ky_all[:, HOLD[:, t]]).reshape(-1)/(2*np.pi)   # |k| in [0,0.5]
        rads.append(np.tile(rr[None, :], (ncc, 1)))
    pf = np.concatenate([p.reshape(-1) for p in preds]); mf = np.concatenate([m.reshape(-1) for m in meass]); rf = np.concatenate([r.reshape(-1) for r in rads])
    s = complex(np.sum(np.conj(pf)*mf)/(np.sum(np.abs(pf)**2)+1e-30))
    def bnmse(mask): return float(np.sum(np.abs(s*pf[mask]-mf[mask])**2)/(np.sum(np.abs(mf[mask])**2)+1e-30))
    r01 = rf/0.5                                                                              # normalize |k| to [0,1]
    inner, mid, outer, allb = bnmse(r01 < 0.15), bnmse((r01 >= 0.15) & (r01 < 0.4)), bnmse(r01 >= 0.4), bnmse(r01 >= 0)
    rows.append((K, inner, mid, outer, allb))
    print(f"[K={K:2d}] held-out NMSE inner(|k|<0.15) {inner:.4e} | mid {mid:.4e} | outer {outer:.4e} | all {allb:.4e}", flush=True)
Kstar_inner = min(rows, key=lambda r: r[1])[0]; Kstar_all = min(rows, key=lambda r: r[4])[0]
print(f"\nREAL CS CV: K* by LOW-|k| (informative, signal>noise) = {Kstar_inner}; K* by global = {Kstar_all}  (reference currently K=5)")
import csv
with open("/net/beegfs/users/P101440/DCE_NIK/results/realdata_nik_vs_cs_figures/real_cs_kcv.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["K", "inner_lowk", "mid", "outer", "all"]); [w.writerow([r[0], f"{r[1]:.6e}", f"{r[2]:.6e}", f"{r[3]:.6e}", f"{r[4]:.6e}"]) for r in rows]
print("REAL_CS_KCV_DONE")
