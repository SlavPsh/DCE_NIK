"""TASK 4C shared pipeline (single source of truth for trainer + eval; no duplicated recon).
Only hidden width and seed vary; everything else frozen to the Task-4 NIK-F0 f25 config."""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, torch
from types import SimpleNamespace
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK"); sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import nik_adapter as A
from train_grasp_nik import build_model
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
from fftc import ifft2c_mri, crop_img
T4 = "/scratch/rnga/vvpshenov/DCE_NIK/results/task4_xcat_nomotion_pilot"
OUT = "/scratch/rnga/vvpshenov/DCE_NIK/results/task4c_nik_capacity_audit"
AIF = "/scratch/rnga/vvpshenov/DCE_NIK/aif_xcat.npz"
FIX = dict(depth=12, w0=62.0, s0=15.0, k_freq=256, k_sigma=2.5, t_freq=32, t_sigma=1.5, coil_embed_dim=8, env=0.75)
CURRENT_W = 512; STEPS = 40000; BATCH = 16384; LR = 1e-5; WD = 3e-3  # frozen Task-4 optimizer (wd is existing, not added)

def load():
    S = np.load(f"{T4}/arrays/sim.npz"); M = np.load(f"{OUT}/arrays/spoke_masks.npz")
    return S, M

_Y100 = None
def _y100():
    global _Y100
    if _Y100 is None: _Y100 = np.load(f"{T4}/arrays/y100.npy", allow_pickle=True)
    return _Y100

def _dataset(S, mask, dev):
    """complete-spoke dataset from y100 (all-9-angle noisy k-space) at `mask` spokes. model coord=2*traj."""
    kx, ky, times, b1 = S["kx"], S["ky"], S["times"], S["b1"]; F, NA, RO = kx.shape; C = b1.shape[-1]; Tt = float(times.max())
    y100 = _y100()
    tn = (2.0 * times / Tt - 1.0).astype(np.float32)
    xs, ys, ts, cs, rr = [], [], [], [], []
    for t in range(F):
        yt = np.asarray(y100[t]).astype(np.complex64).reshape(C, NA, RO)
        for a in range(NA):
            if not mask[t, a]: continue
            kxa = kx[t, a].reshape(-1); kya = ky[t, a].reshape(-1); m = kxa.size
            for c in range(C):
                xs.append(np.stack([2 * kxa, 2 * kya], 1)); ys.append(np.stack([yt[c, a].real, yt[c, a].imag], 1))
                ts.append(np.full(m, tn[t], np.float32)); cs.append(np.full(m, c, np.int64)); rr.append(np.abs(kxa + 1j * kya) / 0.5)
    X = torch.tensor(np.concatenate(xs), dtype=torch.float32, device=dev); Y = torch.tensor(np.concatenate(ys), dtype=torch.float32, device=dev)
    T = torch.tensor(np.concatenate(ts), dtype=torch.float32, device=dev); Ct = torch.tensor(np.concatenate(cs), dtype=torch.long, device=dev)
    R = np.concatenate(rr)
    return X, Y, T, Ct, R, (F, NA, RO, C, Tt)

def build_train(S, dev):
    M = np.load(f"{OUT}/arrays/spoke_masks.npz")
    X, Y, T, Ct, _, dims = _dataset(S, M["train"], dev)
    dcf = compute_dcf_radial(X, method="simple_ramp")
    nz = KSpaceNormalizer(); nz.fit(X, Y, dcf=dcf, envelope_exponent=FIX["env"]); Yn = nz.normalize(X, Y)
    return X, Yn, T, Ct, nz, dims

def make_model(width, seed, ncc, dev):
    torch.manual_seed(seed)
    args = SimpleNamespace(model="wire_ff_patlak", patlak_free=0, aif_file=AIF, hidden=int(width), ff_seed=seed,
        phi_hidden=64, phi_depth=3, phi_w0=30.0, **{k: FIX[k] for k in ["depth","w0","s0","k_freq","k_sigma","t_freq","t_sigma","coil_embed_dim"]})
    return build_model(args, ncc).to(dev)

def param_counts(model):
    tr = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    tot = sum(tr.values())
    coil = sum(v for n, v in tr.items() if "coil_embed" in n)
    temporal = sum(v for n, v in tr.items() if n.split(".")[0] in ("t_grid", "aif_tgrid", "aif_vals", "iaif_vals"))
    spatial = tot - coil - temporal
    return dict(total=tot, spatial=spatial, coil=coil, temporal=temporal)

@torch.no_grad()
def kspace_nmse(model, nz, S, mask, dev, chunk=200000):
    """measured-domain complex k-space NMSE over `mask` spokes + radial shells (0-.3,.3-.7,.7-1)."""
    X, Y, T, Ct, R, _ = _dataset(S, mask, dev)  # Y is measured (un-normalized)
    n = X.shape[0]; num = np.zeros(4); den = np.zeros(4)  # [all, inner, mid, outer]
    def shell(r): return 1 if r < 0.3 else (2 if r < 0.7 else 3)
    for i in range(0, n, chunk):
        pr = nz.denormalize(X[i:i+chunk], model(X[i:i+chunk], T[i:i+chunk], Ct[i:i+chunk]))
        yh = (pr[:, 0] + 1j * pr[:, 1]).cpu().numpy(); yt = (Y[i:i+chunk, 0] + 1j * Y[i:i+chunk, 1]).cpu().numpy()
        rr = R[i:i+chunk]; e = np.abs(yh - yt) ** 2; p = np.abs(yt) ** 2
        num[0] += e.sum(); den[0] += p.sum()
        for sidx in (1, 2, 3):
            m = np.array([shell(r) == sidx for r in rr]); num[sidx] += e[m].sum(); den[sidx] += p[m].sum()
    return dict(nmse=float(num[0]/den[0]), nmse_inner=float(num[1]/den[1]), nmse_mid=float(num[2]/den[2]), nmse_outer=float(num[3]/den[3]))

@torch.no_grad()
def extract_pathC(model, nz, S, dev, chunk=8192):
    """canonical Path C: amplitudes -> denorm -> IFFT per coil -> SENSE common complex maps [N,N,3]."""
    b1 = S["b1"]; N = b1.shape[0]; C = b1.shape[-1]
    coords = torch.from_numpy(A.cartesian_grid(N)).to(dev); P = coords.shape[0]
    rr = np.sqrt(A.cartesian_grid(N)[:, 0]**2 + A.cartesian_grid(N)[:, 1]**2).reshape(N, N)
    cart = np.zeros((N, N, 3, C), np.complex64)
    for c in range(C):
        out = np.zeros((P, 3), np.complex64)
        for i in range(0, P, chunk):
            cco = coords[i:i+chunk]; cc = torch.full((cco.shape[0],), c, dtype=torch.long, device=dev); Amp = model.amplitudes(cco, cc)
            for r in range(3):
                pr = nz.denormalize(cco, Amp[:, r, :].contiguous()); out[i:i+chunk, r] = (pr[:, 0] + 1j * pr[:, 1]).cpu().numpy()
        cart[:, :, :, c] = out.reshape(N, N, 3)
    cart[rr > 1.0] = 0
    a_rc = crop_img(ifft2c_mri(cart.reshape(N, N, 3*C)).reshape(N, N, 3, C), N, N)
    thetaC = np.stack([np.sum(np.conj(b1)*a_rc[:, :, r, :], -1)/(np.sum(np.abs(b1)**2, -1)+1e-8) for r in range(3)], -1)
    return thetaC.astype(np.complex64)

def truth_metrics(thetaC, S):
    """scale-matched (global complex, fit on body) truth metrics. aorta recovery / curve / intAIF / FP energy."""
    labels = S["labels"]; th = S["theta_true"]; Phi = S["Phi"]; times = S["times"]
    body = labels > 0; aorta = labels == 36; Z = body & ~aorta                # exact-zero AIF-truth region
    s = np.vdot(thetaC[body], th[body]) / (np.vdot(thetaC[body], thetaC[body]) + 1e-12)
    tc = thetaC * s
    R_aorta = float(np.abs(tc[aorta][:, 0]).mean() / (np.abs(th[aorta][:, 0]).mean() + 1e-12))
    Ih = np.einsum("xyr,tr->xyt", tc, Phi); Ig = np.einsum("xyr,tr->xyt", th, Phi)
    ca = np.abs(Ih[aorta].mean(0)); cg = np.abs(Ig[aorta].mean(0))
    aorta_curve = float(np.linalg.norm(ca - cg) / (np.linalg.norm(cg) + 1e-12))
    intAIF = float(np.linalg.norm(np.abs(tc[body][:, 1]) - np.abs(th[body][:, 1])) / (np.linalg.norm(np.abs(th[body][:, 1])) + 1e-12))
    E_FP = float(np.linalg.norm(tc[Z][:, 0]) / (np.linalg.norm(th[aorta][:, 0]) + 1e-12))  # spurious AIF energy / true aorta energy
    return dict(R_aorta=R_aorta, aorta_curve_nrmse=aorta_curve, intAIF_nrmse=intAIF, FP_energy=E_FP,
                scale_abs=float(np.abs(s)), scale_phase_deg=float(np.angle(s, deg=True)))
