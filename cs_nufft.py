"""NUFFT-subspace CS: faithful to GRASP-Pro (K=5 navigator-PCA temporal subspace + spatial &
temporal TV) but with a finufft SENSE forward instead of GROG. finufft validated on XCAT (B3).
solve: min ||E x - y||^2 + ls*spatialTV(x) + lt*temporalTV(Phi x), x = K coefficient maps.
E: coeff maps -> per-frame img (x Phi) -> x coil maps -> per-frame NUFFT -> radial samples.
usable on any radial slice (XCAT via xcat_adapter, in-vivo via saved kdc + traj)."""
import numpy as np
import finufft

SIGN = -1.0                                                        # finufft frame vs data (B3)


def temporal_phi(kdc, nline, nt, K=5):
    """K temporal basis from the k-centre navigator. 181x181 TEMPORAL cov (5-readout x ncc
    observations), NOT a coil covariance."""
    nx, nv, ncc = kdc.shape; c0 = nx // 2
    nav = np.abs(kdc[c0 - 2:c0 + 3, :nt * nline, :]).reshape(5, nline, nt, ncc, order="F").mean(1)
    ds = nav.transpose(0, 2, 1).reshape(5 * ncc, nt, order="F")
    w, PC = np.linalg.eigh(np.cov(ds, rowvar=False))
    return PC[:, np.argsort(-w)][:, :K].astype(np.complex64)       # [nt, K]


class NufftSubspace:
    """E and E^H for x [Ny,Nx,K] <-> per-frame radial samples. trajs[t]=(kxf,kyf) normalized."""
    def __init__(self, Phi, b1, trajs, N, eps=1e-5):
        self.Phi = Phi; self.b1 = b1.astype(np.complex64); self.trajs = trajs
        self.N = N; self.eps = eps; self.nt, self.K = Phi.shape; self.C = b1.shape[-1]

    def _co(self, t):
        kx, ky = self.trajs[t]
        return (SIGN * 2 * np.pi * kx.ravel()).astype(np.float64), (SIGN * 2 * np.pi * ky.ravel()).astype(np.float64)

    def fwd(self, x):                                              # x [N,N,K] -> list [C,Mt]
        out = []
        for t in range(self.nt):
            img = (x * self.Phi[t][None, None, :]).sum(-1)         # [N,N]
            cimg = np.ascontiguousarray(np.transpose(img[:, :, None] * self.b1, (2, 0, 1))).astype(np.complex128)
            kx, ky = self._co(t)
            out.append(finufft.nufft2d2(kx, ky, cimg, isign=-1, eps=self.eps))   # [C,Mt]
        return out

    def adj(self, y):                                             # list [C,Mt] -> [N,N,K]
        x = np.zeros((self.N, self.N, self.K), np.complex128)
        for t in range(self.nt):
            kx, ky = self._co(t)
            cimg = finufft.nufft2d1(kx, ky, np.ascontiguousarray(y[t]).astype(np.complex128),
                                    (self.N, self.N), isign=1, eps=self.eps)      # [C,N,N]
            img = (np.conj(self.b1) * np.transpose(cimg, (1, 2, 0))).sum(-1)      # [N,N]
            for k in range(self.K):
                x[:, :, k] += np.conj(self.Phi[t, k]) * img
        return x


def _huber_tv_grad(u, ax, delta=1e-3):
    """gradient of Huber(|d u/d ax|) along axis, edge-preserving L1-ish."""
    d = np.diff(u, axis=ax, append=np.take(u, [-1], axis=ax))
    mag = np.sqrt(np.abs(d) ** 2 + delta ** 2)
    g = d / mag
    return g - np.roll(g, 1, axis=ax)                             # divergence


def recon(E, y, iters=40, ls=1e-3, lt=1e-3, dcf=None, verbose=False):
    """FISTA-lite: data grad via E^H(Ex-y) + spatial(x) & temporal(Phi x) Huber-TV."""
    # good init from density-compensated adjoint
    yi = y if dcf is None else [y[t] * dcf[t][None] for t in range(len(y))]
    x = E.adj(yi)
    x /= (np.abs(x).max() + 1e-12)
    # step size from a couple of power iterations on E^H E
    v = np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
    for _ in range(3):
        v = E.adj(E.fwd(v)); v /= (np.linalg.norm(v) + 1e-12)
    L = np.real(np.vdot(v, E.adj(E.fwd(v)))) + 1e-9
    step = 1.0 / L
    xm = x.copy(); tprev = 1.0
    sc = np.abs(x).max()
    for it in range(iters):
        gdata = E.adj([E.fwd(xm)[t] - y[t] for t in range(E.nt)])
        gtv = np.zeros_like(x)
        for k in range(E.K):
            gtv[:, :, k] += _huber_tv_grad(xm[:, :, k], 0) + _huber_tv_grad(xm[:, :, k], 1)
        # temporal TV on the frame series Phi x (project grad back to subspace)
        img = np.tensordot(xm, E.Phi.T, axes=([2], [0]))          # [N,N,nt]
        gt_img = _huber_tv_grad(img, 2)
        gtv += np.tensordot(gt_img, np.conj(E.Phi), axes=([2], [0])) * lt / max(ls, 1e-9)
        xnew = xm - step * (gdata + ls * sc * gtv)
        t2 = (1 + np.sqrt(1 + 4 * tprev ** 2)) / 2
        xm = xnew + ((tprev - 1) / t2) * (xnew - x)
        x = xnew; tprev = t2
        if verbose and (it % 10 == 0 or it == iters - 1):
            r = sum(np.linalg.norm(E.fwd(x)[t] - y[t]) ** 2 for t in range(E.nt))
            print(f"    iter {it:3d}  data resid {r:.3e}", flush=True)
    return x                                                     # coefficient maps


def frames_from_coeff(x, Phi):
    return np.tensordot(x, Phi.T, axes=([2], [0]))               # [N,N,nt] complex
