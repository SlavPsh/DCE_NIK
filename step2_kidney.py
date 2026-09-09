"""STEP 2: repeat A1 on a kidney-bearing slice. effective rank of the model-free
spatiotemporal dynamics + navigator-K5 residual map. the residual source is model-free
NUFFT (independent of the K=5 CS subspace) so the escape-from-K5 test is not circular.
usage: python step2_kidney.py 19
out: figures/<dt>_step2_slice<z>.png + printed rank/residual summary."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.ndimage as ndi, finufft, json, sys
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
from figpath import fig as fpath

Z = int(sys.argv[1]) if len(sys.argv) > 1 else 19
D = "/net/beegfs/users/P101440/DCE_NIK"; REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"
TA = 375.0

sh = np.load(f"{REF}/shared.npz")
traj = np.asarray(sh["traj_norm"]).astype(np.complex64)
vt = np.asarray(sh["view_time"]).ravel().astype(np.float64)
nx = int(sh["nx"]); bas = int(sh["bas"]); nt = int(sh["nt"])
ftime = np.asarray(sh["frame_time"]).ravel().astype(np.float64)     # [342] CS frame-grid times
sl = np.load(f"{REF}/slice_{Z:02d}.npz")
kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64)
b1 = np.asarray(sl["b1"]).astype(np.complex64); ncc = kdata.shape[2]
den = np.sum(np.abs(b1) ** 2, 2) + 1e-12

# --- navigator K=5 subspace, rebuilt PER-SLICE exactly as recon_slice/A1 (magnitude center
# readout, temporal cov of 5-readout x ncc observations, real modes on the 342 frame grid) ---
c0 = kdata.shape[0] // 2; nline = kdata.shape[1] // nt
nav = np.abs(kdata[c0 - 2:c0 + 3, :nt * nline, :]).reshape(5, nline, nt, ncc, order="F").mean(1)  # (5,nt,ncc)
dseq = nav.transpose(0, 2, 1).reshape(5 * ncc, nt, order="F")       # (5ncc, nt)
w, PC = np.linalg.eigh(np.cov(dseq, rowvar=False))
Phi5_342 = np.real(PC[:, np.argsort(-w)][:, :5]).astype(np.float64) # (nt,5) real navigator basis
meta = json.load(open(f"{D}/results_nufft/meta.json")); SIGN = meta["sign"]; order = np.argsort(vt)
def win_img(idx):
    tr = traj[:, idx]; w = np.maximum(np.abs(tr), 1 / nx / 4)
    x = (SIGN * 2 * np.pi * tr.real).ravel().astype(np.float64); y = (SIGN * 2 * np.pi * tr.imag).ravel().astype(np.float64)
    acc = sum(finufft.nufft2d1(x, y, (kdata[:, idx, c] * w).astype(np.complex128).ravel(), (nx, nx), isign=1, eps=1e-4) * np.conj(b1[:, :, c]) for c in range(ncc))
    s = (nx - bas) // 2; return np.abs(acc / den)[s:s + bas, s:s + bas]
W, STEP = 31, 7
wins = [order[a:a + W] for a in range(0, len(order) - W + 1, STEP)]
tmf = np.array([vt[idx].mean() * TA for idx in wins])
mf = np.stack([win_img(idx) for idx in wins])              # [nwin, bas, bas]
print(f"slice {Z}: model-free windows {mf.shape} (W={W} step={STEP})", flush=True)

# navigator basis interpolated onto the window grid, orthonormalized
P = np.stack([np.interp(tmf, ftime, Phi5_342[:, j]) for j in range(5)], axis=1)  # (nwin,5)
P, _ = np.linalg.qr(P); PtPi = np.linalg.inv(P.T @ P)
def resid_curve(c):                                        # structural residual after K=5 projection
    c = np.atleast_2d(c); proj = (c @ P) @ PtPi @ P.T
    return np.linalg.norm(c - proj, axis=1) / (np.linalg.norm(c - c.mean(1, keepdims=True), axis=1) + 1e-9)

# --- body mask + Gavish-Donoho SVD denoise of the model-free Casorati matrix ---
# raw per-voxel model-free is noise-dominated; the top singular components are real dynamics,
# the tail is streak/noise. optimal hard threshold (unknown noise) separates them.
m = mf.mean(0); body = m > np.quantile(m, 0.55)
Xmat = mf[:, body].T                                       # [nvox, nwin]
mu = Xmat.mean(1, keepdims=True); Xc = Xmat - mu
U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
beta = min(Xc.shape) / max(Xc.shape)
omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43   # Gavish-Donoho 2014, unknown noise
tau = omega * np.median(s); r_sig = int((s > tau).sum())
ev = np.cumsum(s ** 2) / np.sum(s ** 2)
def rank_at(f): return int(np.searchsorted(ev, f) + 1)
r90, r99 = rank_at(0.90), rank_at(0.99)
print(f"signal rank (Gavish-Donoho) r_sig={r_sig}  |  raw spectrum r@90={r90} r@99={r99}  (nwin={mf.shape[0]})")

# denoise to signal subspace, then per-voxel K=5 residual on the denoised curves
Xden = (U[:, :r_sig] * s[:r_sig]) @ Vt[:r_sig] + mu
rs = resid_curve(Xden); rmap = np.zeros_like(m); rmap[body] = rs
r999 = r99
print(f"K5 residual (denoised r={r_sig}): median {np.median(rs):.2f}  p85 {np.percentile(rs,85):.2f}  p95 {np.percentile(rs,95):.2f}  max {rs.max():.2f}")

# --- figure: anatomy, residual map, singular spectrum ---
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
fr = int(np.argmin(np.abs(tmf - 90))); a = mf[fr]
ax[0].imshow(np.rot90(a), cmap="gray", vmax=np.percentile(a, 99.5)); ax[0].axis("off"); ax[0].set_title(f"slice {Z} model-free @{tmf[fr]:.0f}s")
im = ax[1].imshow(np.rot90(rmap), cmap="inferno", vmax=np.percentile(rs, 97)); ax[1].axis("off"); ax[1].set_title("K=5 residual map"); fig.colorbar(im, ax=ax[1], fraction=.046)
ax[2].semilogy(np.arange(1, len(s) + 1), s / s[0], "-o", ms=3)
ax[2].axhline(tau / s[0], color="0.5", ls="-", lw=1, label="GD noise floor")
ax[2].axvline(5, color="r", ls="--", lw=1, label="K=5 (CS)"); ax[2].axvline(r_sig, color="g", ls=":", lw=1.5, label=f"signal rank={r_sig}")
ax[2].set_xlim(0, min(60, len(s))); ax[2].set_xlabel("component"); ax[2].set_ylabel("norm. singular value"); ax[2].grid(alpha=.3); ax[2].legend(); ax[2].set_title("dynamics singular spectrum")
fig.suptitle(f"STEP 2 substrate: slice {Z} effective rank & K=5 residual (model-free)", fontweight="bold")
fig.tight_layout(); p = fpath(f"step2_slice{Z}.png"); fig.savefig(p, dpi=135)
np.savez(f"{D}/step2_slice{Z}.npz", rmap=rmap, body=body, mf=mf.astype(np.float32), tmf=tmf, Phi5=P, sv=s)
print(f"wrote {p.split('/')[-1]} + step2_slice{Z}.npz")
