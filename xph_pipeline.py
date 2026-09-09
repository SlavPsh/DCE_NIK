"""NIK pipeline for the physical no-motion XCAT (shared by trainer + eval). F0 only. Mirrors the
frozen Task-4C recipe; only hidden width, k_sigma (spatial bandwidth) and seed vary."""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, torch
from types import SimpleNamespace
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
import nik_adapter as A, xph_common as X
from train_grasp_nik import build_model
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
from fftc import ifft2c_mri, crop_img
AIF = "/net/beegfs/users/P101440/DCE_NIK/aif_xph.npz"
OUT = X.OUT; ZI = 15
OVERSAMPLE = 2   # k-space grid oversampling for the image render: single-FOV iFFT of the coordinate
#                network aliases (L2/L3); query on OVERSAMPLE*RO grid, iFFT, crop central RO. L3 gate:
#                +1.7..+5.3 dB PSNR / +0.08..+0.13 HaarPSI vs truth. Set 1 to restore the old render.
FIX = dict(depth=12, w0=62.0, s0=15.0, k_freq=256, k_sigma=2.5, t_freq=32, t_sigma=1.5, coil_embed_dim=8, env=0.75)
STEPS = 40000; BATCH = 16384; LR = 1e-5; WD = 3e-3
# spoke split: 7 angles/frame -> train {0-4} (recon input, 5 spokes/frame = Proj5), val {5}, test {6}
TRAIN_ANG, VAL_ANG, TEST_ANG = [0, 1, 2, 3, 4], [5], [6]

_D = None
def data():
    global _D
    if _D is None: _D = X.load_slice(ZI)
    return _D

def masks():
    d = data(); F, nang = d["kx"].shape[:2]
    m = {"train": np.zeros((F, nang), bool), "val": np.zeros((F, nang), bool), "test": np.zeros((F, nang), bool)}
    m["train"][:, TRAIN_ANG] = True; m["val"][:, VAL_ANG] = True; m["test"][:, TEST_ANG] = True
    return m

def _dataset(mask, dev):
    d = data(); kx, ky, times = d["kx"], d["ky"], d["times"]; F, nang, RO = kx.shape; C = d["b1"].shape[-1]; Tt = float(times.max())
    tn = (2.0 * times / Tt - 1.0).astype(np.float32)
    xs, ys, ts, cs, rr = [], [], [], [], []
    for t in range(F):
        for a in range(nang):
            if not mask[t, a]: continue
            kxa = kx[t, a]; kya = ky[t, a]; m = kxa.size
            for c in range(C):
                xs.append(np.stack([2 * kxa, 2 * kya], 1)); yv = d["kdata"][c, t, a]
                ys.append(np.stack([yv.real, yv.imag], 1)); ts.append(np.full(m, tn[t], np.float32))
                cs.append(np.full(m, c, np.int64)); rr.append(np.abs(kxa + 1j * kya) / 0.5)
    X_ = torch.tensor(np.concatenate(xs), dtype=torch.float32, device=dev); Y_ = torch.tensor(np.concatenate(ys), dtype=torch.float32, device=dev)
    T_ = torch.tensor(np.concatenate(ts), dtype=torch.float32, device=dev); C_ = torch.tensor(np.concatenate(cs), dtype=torch.long, device=dev)
    return X_, Y_, T_, C_, np.concatenate(rr), (F, nang, RO, C, Tt)

def build_train(dev, env=None):
    env = FIX["env"] if env is None else float(env)                     # envelope_exponent (D4 sweep); default 0.75
    X_, Y_, T_, C_, _, dims = _dataset(masks()["train"], dev)
    dcf = compute_dcf_radial(X_, method="simple_ramp")
    nz = KSpaceNormalizer(); nz.fit(X_, Y_, dcf=dcf, envelope_exponent=env); Yn = nz.normalize(X_, Y_)
    return X_, Yn, T_, C_, nz, dims

def make_model(width, k_sigma, seed, ncc, dev):
    torch.manual_seed(seed)
    args = SimpleNamespace(model="wire_ff_patlak", patlak_free=0, aif_file=AIF, hidden=int(width), ff_seed=seed,
        phi_hidden=64, phi_depth=3, phi_w0=30.0, k_sigma=float(k_sigma),
        **{k: FIX[k] for k in ["depth", "w0", "s0", "k_freq", "t_freq", "t_sigma", "coil_embed_dim"]})
    return build_model(args, ncc).to(dev)

TOFTS_BASIS = "/net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/basis_xph.npz"
def make_model_tofts(width, k_sigma, seed, ncc, dev, basis_file=None):
    """nik_tofts_subspace: same backbone/FIX as make_model (F0), fixed ext-Tofts atoms instead of Patlak."""
    torch.manual_seed(seed)
    args = SimpleNamespace(model="wire_ff_tofts", tofts_basis=basis_file or TOFTS_BASIS, hidden=int(width), ff_seed=seed,
        k_sigma=float(k_sigma), **{k: FIX[k] for k in ["depth", "w0", "s0", "k_freq", "t_freq", "t_sigma", "coil_embed_dim"]})
    return build_model(args, ncc).to(dev)

def make_model_g(model_type, width, k_sigma, seed, ncc, dev, rank=5, warmstart=False):
    """generalized builder: wire_ff_patlak (F0), wire_ff_subspace (free rank-R), wire_ff (free continuous)."""
    torch.manual_seed(seed)
    args = SimpleNamespace(model=model_type, patlak_free=0, aif_file=AIF, hidden=int(width), ff_seed=seed,
        rank=int(rank), phi_hidden=64, phi_depth=3, phi_w0=30.0, phi_ortho=False, n_pk=-1, radial_alpha=1.0,
        k_sigma=float(k_sigma), **{k: FIX[k] for k in ["depth", "w0", "s0", "k_freq", "t_freq", "t_sigma", "coil_embed_dim"]})
    m = build_model(args, ncc).to(dev)
    if model_type == "wire_ff_subspace" and warmstart:
        from nik_model import warmstart_phi
        Phi = build_phi_pca(rank)                                        # [F,rank] complex (k-centre PCA, same as GRASP)
        Tt = float(data()["times"].max()); ft = (2 * data()["times"] / Tt - 1).astype(np.float32)
        err = warmstart_phi(m, ft, Phi, steps=800, device=str(dev)); print(f"  warmstart Phi<-PCA (rank {rank}) MSE {err:.3e}", flush=True)
    return m

def build_phi_pca(K=5):
    """K temporal PCA components from the k-space-centre navigator (same as GRASP-Pro's basis)."""
    d = data(); RO = d["kdata"].shape[-1]; c0 = RO // 2
    nav = np.abs(d["kdata"][:, :, :, c0-2:c0+3]); F = nav.shape[1]
    ds = nav.transpose(0, 2, 3, 1).reshape(-1, F)
    w, PC = np.linalg.eigh(np.cov(ds, rowvar=False))
    return PC[:, np.argsort(-w)][:, :K].astype(np.complex64)

@torch.no_grad()
def extract_coeffs_g(model, nz, dev, chunk=8192):
    """general Path C: rank-R complex coefficient maps [RO,RO,R] (R read from the model). None if no subspace."""
    if not hasattr(model, "amplitudes"): return None
    d = data(); b1 = d["b1"]; RO = b1.shape[0]; C = b1.shape[-1]; NG = OVERSAMPLE * RO
    coords = torch.from_numpy(A.cartesian_grid(NG)).to(dev); P_ = coords.shape[0]
    rr = np.sqrt(A.cartesian_grid(NG)[:, 0]**2 + A.cartesian_grid(NG)[:, 1]**2).reshape(NG, NG)
    Rr = model.amplitudes(coords[:2], torch.zeros(2, dtype=torch.long, device=dev)).shape[1]
    cart = np.zeros((NG, NG, Rr, C), np.complex64)
    for c in range(C):
        out = np.zeros((P_, Rr), np.complex64)
        for i in range(0, P_, chunk):
            cc = coords[i:i+chunk]; cd = torch.full((cc.shape[0],), c, dtype=torch.long, device=dev); Amp = model.amplitudes(cc, cd)
            for r in range(Rr):
                pr = nz.denormalize(cc, Amp[:, r, :].contiguous()); out[i:i+chunk, r] = (pr[:, 0] + 1j * pr[:, 1]).cpu().numpy()
        cart[:, :, :, c] = out.reshape(NG, NG, Rr)
    cart[rr > 1.0] = 0
    a_rc = crop_img(ifft2c_mri(cart.reshape(NG, NG, Rr*C)).reshape(NG, NG, Rr, C), RO, RO)
    return np.stack([np.sum(np.conj(b1)*a_rc[:, :, r, :], -1)/(np.sum(np.abs(b1)**2, -1)+1e-8) for r in range(Rr)], -1).astype(np.complex64)

@torch.no_grad()
def reconstruct_g(model, nz, t_query_s, dev):
    """general NIK dynamic: subspace/patlak via coeffs@basis (fast); free wire_ff via per-time query."""
    if hasattr(model, "amplitudes"):
        thC = extract_coeffs_g(model, nz, dev)
        Tt = float(data()["times"].max()); tn = torch.tensor([2*t/Tt-1 for t in t_query_s], dtype=torch.float32, device=dev)
        Phi = model.basis(tn).cpu().numpy()[:, :, 0].astype(np.float32)  # [T,R]
        return np.einsum("xyr,tr->xyt", thC, Phi).astype(np.complex64)
    return reconstruct(model, nz, t_query_s, dev)                        # free model: per-time

def param_counts(model):
    tr = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    return dict(total=sum(tr.values()), spatial=sum(v for n, v in tr.items() if n.split(".")[0] not in ("t_grid", "aif_tgrid", "aif_vals", "iaif_vals", "coil_embed")),
                coil=sum(v for n, v in tr.items() if "coil_embed" in n))

@torch.no_grad()
def kspace_nmse(model, nz, mask, dev, chunk=200000):
    X_, Y_, T_, C_, R, _ = _dataset(mask, dev); n = X_.shape[0]; num = np.zeros(4); den = np.zeros(4)
    sh = lambda r: 1 if r < 0.3 else (2 if r < 0.7 else 3)
    for i in range(0, n, chunk):
        pr = nz.denormalize(X_[i:i+chunk], model(X_[i:i+chunk], T_[i:i+chunk], C_[i:i+chunk]))
        yh = (pr[:, 0] + 1j * pr[:, 1]).cpu().numpy(); yt = (Y_[i:i+chunk, 0] + 1j * Y_[i:i+chunk, 1]).cpu().numpy()
        rr = R[i:i+chunk]; e = np.abs(yh - yt)**2; p = np.abs(yt)**2; num[0] += e.sum(); den[0] += p.sum()
        for s in (1, 2, 3):
            mk = np.array([sh(r) == s for r in rr]); num[s] += e[mk].sum(); den[s] += p[mk].sum()
    return dict(nmse=float(num[0]/den[0]), inner=float(num[1]/den[1]), mid=float(num[2]/den[2]), outer=float(num[3]/den[3]))

@torch.no_grad()
def extract_pathC(model, nz, dev, chunk=8192):
    """canonical Path C coefficient maps [RO,RO,R] (complex, SENSE-combined). R = model.rank (3 for Patlak)."""
    d = data(); b1 = d["b1"]; RO = b1.shape[0]; C = b1.shape[-1]; NG = OVERSAMPLE * RO; Rk = int(getattr(model, "rank", 3))
    coords = torch.from_numpy(A.cartesian_grid(NG)).to(dev); P_ = coords.shape[0]
    rr = np.sqrt(A.cartesian_grid(NG)[:, 0]**2 + A.cartesian_grid(NG)[:, 1]**2).reshape(NG, NG)
    cart = np.zeros((NG, NG, Rk, C), np.complex64)
    for c in range(C):
        out = np.zeros((P_, Rk), np.complex64)
        for i in range(0, P_, chunk):
            cc = coords[i:i+chunk]; cd = torch.full((cc.shape[0],), c, dtype=torch.long, device=dev); Amp = model.amplitudes(cc, cd)
            for r in range(Rk):
                pr = nz.denormalize(cc, Amp[:, r, :].contiguous()); out[i:i+chunk, r] = (pr[:, 0] + 1j * pr[:, 1]).cpu().numpy()
        cart[:, :, :, c] = out.reshape(NG, NG, Rk)
    cart[rr > 1.0] = 0
    a_rc = crop_img(ifft2c_mri(cart.reshape(NG, NG, Rk*C)).reshape(NG, NG, Rk, C), RO, RO)
    return np.stack([np.sum(np.conj(b1)*a_rc[:, :, r, :], -1)/(np.sum(np.abs(b1)**2, -1)+1e-8) for r in range(Rk)], -1).astype(np.complex64)

@torch.no_grad()
def basis_at(model, t_query_s, dev):
    """fixed-basis Phi(t) [len(t),R] at physical query times (F0 R=3, tofts R=rank). real channel."""
    Tt = float(data()["times"].max()); tn = torch.tensor([2*t/Tt-1 for t in t_query_s], dtype=torch.float32, device=dev)
    return model.basis(tn).cpu().numpy()[:, :, 0].astype(np.float32)   # real channel -> [len(t),R]

@torch.no_grad()
def reconstruct_pathC(model, nz, t_query_s, dev):
    """NIK-F0 dynamic via Path C: theta_C @ Phi(t). complex [RO,RO,len(t)] (rot180 NOT applied)."""
    thC = extract_pathC(model, nz, dev); Phi = basis_at(model, t_query_s, dev)
    return np.einsum("xyr,tr->xyt", thC, Phi).astype(np.complex64), thC

@torch.no_grad()
def reconstruct(model, nz, t_query_s, dev, chunk=8192):
    """NIK dynamic (complex coil-combined signal) at physical query times [s] -> [RO,RO,len(t)]."""
    d = data(); b1 = d["b1"]; RO = b1.shape[0]; C = b1.shape[-1]; Tt = float(d["times"].max()); NG = OVERSAMPLE * RO
    coords = torch.from_numpy(A.cartesian_grid(NG)).to(dev); P = coords.shape[0]
    rr = np.sqrt(A.cartesian_grid(NG)[:, 0]**2 + A.cartesian_grid(NG)[:, 1]**2).reshape(NG, NG)
    out = np.zeros((RO, RO, len(t_query_s)), np.complex64)
    for j, ts in enumerate(t_query_s):
        tn = float(2 * ts / Tt - 1); ci = np.zeros((NG, NG, C), np.complex64)
        for c in range(C):
            v = np.zeros(P, np.complex64)
            for i in range(0, P, chunk):
                cc = coords[i:i+chunk]; tt = torch.full((cc.shape[0],), tn, device=dev); cd = torch.full((cc.shape[0],), c, dtype=torch.long, device=dev)
                pr = nz.denormalize(cc, model(cc, tt, cd)); v[i:i+chunk] = (pr[:, 0] + 1j * pr[:, 1]).cpu().numpy()
            ci[:, :, c] = v.reshape(NG, NG)
        ci[rr > 1.0] = 0; im = crop_img(ifft2c_mri(ci), RO, RO); out[:, :, j] = np.sum(np.conj(b1) * im, -1) / (np.sum(np.abs(b1)**2, -1) + 1e-8)
    return out
