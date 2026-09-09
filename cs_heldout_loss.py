"""CS-70 / CS-100 held-out k-space loss on the SAME held-out spokes NIK validates on,
in the SAME NIK-normalized space -> directly comparable to NIK's held-out MSE.

CS is reconstructed (complex, coil-resolved via SENSE b1), forward-projected to the radial
held-out coords with finufft (SENSE forward), LS-scaled to the measured data on TRAIN spokes,
then normalized with NIK's KSpaceNormalizer. A validation gate checks the forward operator
reproduces measured data on TRAIN spokes before trusting held-out numbers.
"""
import sys, numpy as np, torch, finufft
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
import nik_adapter as NA
from kspace_normalization import compute_dcf_radial, KSpaceNormalizer
from grog import get_gx_gy, grog_dictionary_interp
from operators import TempPCASub, Emat_GROG2Dksp, TV_Temp, FD1OP
from cs_solver import cs_l1_nlcg_sptv
from fftc import ifft2c_mri
from coilmaps import adapt_array_2d
import precompute_ref as pr

OUT = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; SLC = 13
NT_CS = 34                                        # CS temporal binning for the recon
pr.Weight1 = 0.001; pr.Weight2 = 0.0005

# ---------- data + the exact NIK 70/30 split ----------
sh = NA.load_shared(OUT); sl = NA.load_slice(OUT, SLC)
krad = np.asarray(sl["kdata_radial"]); traj = np.asarray(sh["traj_norm"]); vt = np.asarray(sh["view_time"]).ravel()
nx, ntv, ncc = krad.shape
ds = NA.make_radial_dataset(OUT, SLC, compute_device="cpu", shared=sh)
x, t, c, y_raw, sid = ds["x_all"], ds["t_all"], ds["coil_all"], ds["y_all_raw"], ds["spoke_id_all"]
uniq = torch.unique(sid); n_tr = max(1, int(uniq.numel() * 0.7))
g = torch.Generator().manual_seed(0)
perm = uniq[torch.randperm(uniq.numel(), generator=g)]
train_v = perm[:n_tr].sort().values.numpy(); held_v = perm[n_tr:].sort().values.numpy()
he_mask = torch.isin(sid, perm[n_tr:]); tr_mask = ~he_mask
print(f"views: {ntv} total, {len(train_v)} train, {len(held_v)} held-out", flush=True)

# NIK normalizer (fit on train samples, env 0.75) -- the exact space NIK's held-out lives in
dcf = compute_dcf_radial(x, method="simple_ramp")
nz = KSpaceNormalizer(); nz.fit(x[tr_mask], y_raw[tr_mask], dcf=dcf[tr_mask], envelope_exponent=0.75)
y_norm = nz.normalize(x, y_raw)                    # (N,2) normalized measured

# ---------- CS recon (complex, coil-resolved) on a spoke subset ----------
def cs_complex(views):
    kd = krad[:, views, :]; Tg = np.asarray(sh["traj_grog"])[:, views]
    nsp = len(views); nl = nsp // NT_CS; use = nl * NT_CS
    c0 = kd.shape[0] // 2; nc = kd.shape[2]
    nav = kd[c0 - 2:c0 + 3, :use, :].reshape(5, nl, NT_CS, nc, order="F").mean(1)
    dsq = np.abs(nav).transpose(0, 2, 1).reshape(5 * nc, NT_CS, order="F")
    w, PC = np.linalg.eigh(np.cov(dsq, rowvar=False)); Phi = PC[:, np.argsort(-w)][:, :5].astype(np.complex64)
    pr.K = 5; pr.ncc = nc
    D = kd.reshape(nx * nsp, nc, order="F"); U, S, Vh = np.linalg.svd(D, full_matrices=False)
    Vcc = (Vh.conj().T)[:, :nc]; kdc = (D @ Vcc).reshape(nx, nsp, nc, order="F").astype(np.complex64)
    kk = kdc[:, :, None, :]; Gx, Gy = get_gx_gy(kk, Tg)
    kref, _ = grog_dictionary_interp(kk, Gx, Gy, Tg, 1); b1 = adapt_array_2d(np.squeeze(ifft2c_mri(kref)))
    b1 = (b1 / np.abs(b1).max()).astype(np.complex64)
    _, DCF = grog_dictionary_interp(kk[:, -pr.Nqu:], Gx, Gy, Tg[:, -pr.Nqu:], 0)
    kd2 = kdc[:, :use, :].reshape(nx, nl, NT_CS, nc, order="F"); Tr2 = Tg[:, :use].reshape(nx, nl, NT_CS, order="F")
    kd3, DCF_U = grog_dictionary_interp(kd2, Gx, Gy, Tr2, 1); mask = (kd3[:, :, :, 0] != 0).astype(np.complex64)
    Wc0 = np.repeat(DCF, NT_CS, 2) / DCF_U; PCA = TempPCASub(Phi)
    kdatac = np.stack([PCA @ kd3[:, :, :, ii] for ii in range(nc)], 3)
    Wc = PCA @ Wc0; Wc = np.repeat(Wc[:, :, :1], 5, 2)[:, :, :, None]; Wc = np.repeat(Wc, nc, 3)
    E = Emat_GROG2Dksp(mask, b1, Wc, PCA, 1); yv = (kdatac * np.sqrt(Wc)).astype(np.complex64); recon = E.H @ yv
    p = dict(E=E, y=yv, PCA=PCA, TV1=TV_Temp(), TV2=FD1OP(),
             TVWeight1=np.abs(recon).max() * pr.Weight1, TVWeight2=np.abs(recon).max() * pr.Weight2, nite=5)
    for _ in range(3): recon = cs_l1_nlcg_sptv(recon, p)
    Xf = (PCA.H @ recon).astype(np.complex64)      # [skx,skx,NT_CS] complex coil-combined image
    ft = np.array([vt[views[f * nl:(f + 1) * nl]].mean() for f in range(NT_CS)])
    return Xf, b1, ft                              # b1 [skx,skx,nc]

# ---------- SENSE forward-projection to radial coords (finufft type-2) ----------
def forward(Xf, b1, ft_cs, conv, only_frames=None):
    isign, shift = conv
    kxr = (2 * np.pi * np.real(traj)).astype(np.float64)          # [nx,ntv] radians, Nyq at traj=0.5
    kyr = (2 * np.pi * np.imag(traj)).astype(np.float64)
    fr = np.argmin(np.abs(vt[None, :] - ft_cs[:, None]), axis=0)  # view -> nearest CS frame
    ypred = np.zeros((nx, ntv, ncc), np.complex64)
    frames = range(len(ft_cs)) if only_frames is None else only_frames
    for f in frames:
        vsel = np.where(fr == f)[0]
        if not len(vsel): continue
        kx = kxr[:, vsel].reshape(-1); ky = kyr[:, vsel].reshape(-1)
        for cc in range(ncc):
            img = (b1[:, :, cc] * Xf[:, :, f]).astype(np.complex128)
            if shift: img = np.fft.ifftshift(img)
            vals = finufft.nufft2d2(kx, ky, img, isign=isign, eps=1e-6)
            ypred[:, vsel, cc] = vals.reshape(nx, len(vsel))
    return ypred

def to_samples(yc):                                # [nx,ntv,ncc] -> (N,2) in nik_adapter (c,r,v) order
    yv = np.transpose(yc, (2, 0, 1)).reshape(-1); return np.stack([yv.real, yv.imag], 1).astype(np.float32)

y_meas_c = np.transpose(krad, (2, 0, 1)).reshape(ncc, nx, ntv)    # measured, (c,r,v)
tr_s = tr_mask.numpy(); he_s = he_mask.numpy()

CONV = None
for tag, views in [("CS-100", np.arange(ntv)), ("CS-70", train_v)]:
    Xf, b1, ft_cs = cs_complex(views)
    if CONV is None:                               # detect convention ONCE, on 3 probe frames (cheap)
        probe = sorted({0, len(ft_cs) // 2, len(ft_cs) - 1}); best = None
        fr = np.argmin(np.abs(vt[None, :] - ft_cs[:, None]), axis=0)
        probe_views = np.where(np.isin(fr, probe))[0]
        m = tr_s & np.isin(sid.numpy(), probe_views)               # train samples in probe frames
        for conv in [(-1, True), (1, True), (-1, False), (1, False)]:
            yp = forward(Xf, b1, ft_cs, conv, only_frames=probe); ys = to_samples(yp)
            a, b = ys[m], y_raw.numpy()[m]
            corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            if best is None or corr > best[1]: best = (conv, corr)
        CONV = best[0]; print(f"convention probe -> {CONV} (train-corr {best[1]:.3f})", flush=True)
    yp = forward(Xf, b1, ft_cs, CONV); ys = to_samples(yp)
    a, b = ys[tr_s], y_raw.numpy()[tr_s]
    s = (a * b).sum() / ((a * a).sum() + 1e-30)
    corr = np.corrcoef((s * a).ravel(), b.ravel())[0, 1]
    yp_norm = nz.normalize(x, torch.from_numpy(s * ys))
    he_mse = float(((yp_norm[he_s] - y_norm[he_s]) ** 2).mean())
    tr_mse = float(((yp_norm[tr_s] - y_norm[tr_s]) ** 2).mean())
    print(f"{tag}: train-corr={corr:.3f} (validation)  |  held-out MSE={he_mse:.4f}  train MSE={tr_mse:.4f}", flush=True)

print("\nNIK held-out (same space): full-rank 0.325 | R=5 0.199 | R=10 0.189 | R=20 0.182")
