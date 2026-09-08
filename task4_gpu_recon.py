"""TASK 4 non-neural recons on GPU (cufinufft) - the 4-core node starves CPU finufft (1 iter / 1.6h).
cufinufft matches finufft to 3e-6 (verified), so the direct-discrete stays forward-matched to the
simulation. Mirrors cs_nufft.NufftSubspace + FISTA (SIGN=-1, coords x2pi, Huber-TV) exactly, in cupy.
Produces direct-discrete, CS-reconstruct-then-fit (identity basis), and NUFFT-sanity for f25 + f100.
usage: python task4_gpu_recon.py [--frac f25|f100|both]"""
import warnings; warnings.filterwarnings("ignore")
import argparse, sys, time, numpy as np, cupy as cp, cufinufft
OUT = "/scratch/rnga/vvpshenov/DCE_NIK/results/task4_xcat_nomotion_pilot"; SIGN = -1.0
ap = argparse.ArgumentParser(); ap.add_argument("--frac", default="both", choices=["f25", "f100", "both"]); A = ap.parse_args()
S = np.load(f"{OUT}/arrays/sim.npz"); Phi0 = S["Phi"]; b1n = S["b1"]; kx = S["kx"]; ky = S["ky"]; keep = S["keep_f25"]
F, NA, RO = kx.shape; C = b1n.shape[-1]; N = RO; EPS = 1e-5

class GPUSub:
    """cupy/cufinufft version of cs_nufft.NufftSubspace (identical convention). Two PERSISTENT plans
    (type1/type2, n_trans=C) reused across all frames+iterations via setpts; coords preloaded to GPU."""
    def __init__(self, Phi, b1, trajs):
        self.Phi = cp.asarray(Phi, cp.complex128); self.b1 = cp.asarray(b1, cp.complex128)
        self.nt, self.K = Phi.shape
        self.co = [(cp.asarray(SIGN * 2 * np.pi * kxf.ravel(), cp.float64),
                    cp.asarray(SIGN * 2 * np.pi * kyf.ravel(), cp.float64)) for (kxf, kyf) in trajs]
        self.p2 = cufinufft.Plan(2, (N, N), C, eps=EPS, isign=-1, dtype="complex128")   # img->samples
        self.p1 = cufinufft.Plan(1, (N, N), C, eps=EPS, isign=1, dtype="complex128")    # samples->img
    def fwd(self, x):
        out = []
        for t in range(self.nt):
            img = (x * self.Phi[t][None, None, :]).sum(-1)                 # [N,N]
            cimg = cp.ascontiguousarray(cp.transpose(img[:, :, None] * self.b1, (2, 0, 1)))  # [C,N,N]
            self.p2.setpts(self.co[t][0], self.co[t][1])
            out.append(self.p2.execute(cimg))                             # [C,M]
        return out
    def adj(self, y):
        x = cp.zeros((N, N, self.K), cp.complex128)
        for t in range(self.nt):
            self.p1.setpts(self.co[t][0], self.co[t][1])
            cimg = self.p1.execute(cp.ascontiguousarray(y[t]))            # [C,N,N]
            img = (cp.conj(self.b1) * cp.transpose(cimg, (1, 2, 0))).sum(-1)     # [N,N]
            for k in range(self.K): x[:, :, k] += cp.conj(self.Phi[t, k]) * img
        return x

def htv(u, ax, delta=1e-3):
    last = cp.take(u, cp.array([u.shape[ax] - 1]), axis=ax)
    d = cp.diff(u, axis=ax, append=last); mag = cp.sqrt(cp.abs(d) ** 2 + delta ** 2); g = d / mag
    return g - cp.roll(g, 1, axis=ax)

def recon(E, y, iters=40, ls=1e-3, lt=1e-3, label=""):
    yl = [cp.asarray(np.asarray(y[t]).astype(np.complex128)) for t in range(E.nt)]
    x = E.adj(yl); x /= (cp.abs(x).max() + 1e-12)
    v = cp.asarray(np.random.randn(N, N, E.K) + 1j * np.random.randn(N, N, E.K))
    for _ in range(3): v = E.adj(E.fwd(v)); v /= (cp.linalg.norm(v) + 1e-12)
    L = cp.real(cp.vdot(v, E.adj(E.fwd(v)))) + 1e-9; step = 1.0 / L
    xm = x.copy(); tprev = 1.0; sc = cp.abs(x).max(); Phit = E.Phi.T
    for it in range(iters):
        gdata = E.adj([E.fwd(xm)[t] - yl[t] for t in range(E.nt)])
        gtv = cp.zeros_like(x)
        for k in range(E.K): gtv[:, :, k] += htv(xm[:, :, k], 0) + htv(xm[:, :, k], 1)
        img = cp.tensordot(xm, Phit, axes=([2], [0]))
        gtv += cp.tensordot(htv(img, 2), cp.conj(E.Phi), axes=([2], [0])) * lt / max(ls, 1e-9)
        xnew = xm - step * (gdata + ls * sc * gtv)
        t2 = (1 + np.sqrt(1 + 4 * tprev ** 2)) / 2; xm = xnew + ((tprev - 1) / t2) * (xnew - x); x = xnew; tprev = t2
        if it % 10 == 0 or it == iters - 1:
            r = float(sum(cp.linalg.norm(E.fwd(x)[t] - yl[t]) ** 2 for t in range(E.nt)))
            print(f"    {label} iter {it:3d} resid {r:.3e}", flush=True)
    return cp.asnumpy(x)

def run_frac(frac):
    y = np.load(f"{OUT}/arrays/y{'100' if frac=='f100' else '25_in'}.npy", allow_pickle=True)
    trajs = [(kx[t], ky[t]) for t in range(F)] if frac == "f100" else [(kx[t][keep[t]], ky[t][keep[t]]) for t in range(F)]
    yl = [np.asarray(y[t]).astype(np.complex64) for t in range(F)]
    t0 = time.time()
    # direct-discrete (F0 basis)
    Ed = GPUSub(Phi0, b1n, trajs); thd = recon(Ed, yl, label=f"direct/{frac}")
    np.savez(f"{OUT}/arrays/direct_{frac}.npz", theta_direct=thd.astype(np.complex64)); print(f"[{frac}] direct done {time.time()-t0:.0f}s", flush=True)
    # CS reconstruct-then-fit (identity basis -> free frames -> fit Phi0)
    Ei = GPUSub(np.eye(F).astype(np.complex64), b1n, trajs); xser = recon(Ei, yl, label=f"csfit/{frac}")
    thc = np.einsum("rt,xyt->xyr", np.linalg.pinv(Phi0), xser)
    np.savez(f"{OUT}/arrays/csfit_{frac}.npz", theta_csfit=thc.astype(np.complex64), x_series=xser.astype(np.complex64)); print(f"[{frac}] csfit done {time.time()-t0:.0f}s", flush=True)
    # NUFFT-sanity (dcf-adjoint per frame -> project onto Phi0)
    Icn = np.zeros((N, N, F), np.complex64)
    for t in range(F):
        kxf, kyf = trajs[t]; kxc = cp.asarray(SIGN * 2 * np.pi * kxf.ravel(), cp.float64); kyc = cp.asarray(SIGN * 2 * np.pi * kyf.ravel(), cp.float64)
        w = cp.asarray(np.maximum(np.abs(kxf.ravel() + 1j * kyf.ravel()), 1e-3))
        yc = cp.asarray(np.asarray(yl[t]).reshape(C, -1).astype(np.complex128)) * w[None, :]
        cimg = cufinufft.nufft2d1(kxc, kyc, cp.ascontiguousarray(yc), (N, N), isign=1, eps=EPS)
        img = (cp.conj(cp.asarray(b1n, cp.complex128)) * cp.transpose(cimg, (1, 2, 0))).sum(-1)
        Icn[:, :, t] = cp.asnumpy(img)
    thn = np.einsum("rt,xyt->xyr", np.linalg.pinv(Phi0), Icn)
    np.savez(f"{OUT}/arrays/nufft_{frac}.npz", theta_nufft=thn.astype(np.complex64), Ic=Icn.astype(np.complex64)); print(f"[{frac}] nufft done {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    for frac in (["f25", "f100"] if A.frac == "both" else [A.frac]): run_frac(frac)
    print("GPU_RECON_DONE", flush=True)
