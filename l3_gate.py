"""L3 gate: does the 2x-oversampled render improve the image vs TRUTH (not vs the network)? Render
sub16 + free (all seeds) at oversampling 1x and 2x, score body-masked PSNR/SSIM/HaarPSI/NRMSE vs XCAT
truth, side by side. GATE: material truth improvement -> implement+re-baseline; none -> correctness fix
only, spatial line exhausted. k-space self-consistency is NOT the objective. No training; truth eval-only."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob
import xph_pipeline as P, xph_common as X, nik_adapter as A, recon_asserts as RA
from fftc import ifft2c_mri, crop_img
from masked_metrics import haarpsi_masked, ssim_masked
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); RO = d["b1"].shape[0]; F = len(tq)
b1 = d["b1"].astype(np.complex128); den = np.sum(np.abs(b1)**2, -1)+1e-8; rv = float(Tr[body].max()-Tr[body].min()); Ttime = float(tq.max())
rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1)); FR = np.arange(0, F, 4)

def load(kind, s, st):
    tag = f"{'sub16' if kind=='sub16' else 'free'}_w768_s{s}"; p = f"{P.OUT}/checkpoints/{tag}/ck_{st}.pt"
    if not os.path.exists(p): p = sorted(glob.glob(f"{P.OUT}/checkpoints/{tag}/ck_*.pt"))[-1]
    mm = P.make_model_g("wire_ff_subspace" if kind == "sub16" else "wire_ff", 768, P.FIX["k_sigma"], 0, C, dev, rank=16, warmstart=False)
    mm.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); mm.eval(); return mm

@torch.no_grad()
def render_sub(m, factor, chunk=200000):
    N = factor*RO; grid = torch.from_numpy(A.cartesian_grid(N)).to(dev)
    rrg = np.sqrt(A.cartesian_grid(N)[:, 0]**2+A.cartesian_grid(N)[:, 1]**2).reshape(N, N)
    tn = torch.tensor([2*tq[t]/Ttime-1 for t in range(F)], dtype=torch.float32, device=dev); Phi = m.basis(tn).cpu().numpy()[:, :, 0].astype(np.complex64)
    Rr = Phi.shape[1]; cart = np.zeros((N, N, Rr, C), np.complex64)
    for c in range(C):
        out = np.zeros((grid.shape[0], Rr), np.complex64)
        for i in range(0, grid.shape[0], chunk):
            cc = grid[i:i+chunk]; Amp = m.amplitudes(cc, torch.full((cc.shape[0],), c, dtype=torch.long, device=dev))
            for r in range(Rr): pr = nz.denormalize(cc, Amp[:, r, :].contiguous()); out[i:i+chunk, r] = (pr[:, 0]+1j*pr[:, 1]).cpu().numpy()
        cart[:, :, :, c] = out.reshape(N, N, Rr)
    cart[rrg > 1.0] = 0
    a_rc = crop_img(ifft2c_mri(cart.reshape(N, N, Rr*C)).reshape(N, N, Rr, C), RO, RO)
    thC = np.stack([np.sum(np.conj(b1)*a_rc[:, :, r, :], -1)/den for r in range(Rr)], -1).astype(np.complex64)
    return np.einsum("xyr,tr->xyt", thC, Phi).astype(np.complex64)

@torch.no_grad()
def render_free(m, factor, frames, chunk=200000):
    N = factor*RO; grid = torch.from_numpy(A.cartesian_grid(N)).to(dev)
    rrg = np.sqrt(A.cartesian_grid(N)[:, 0]**2+A.cartesian_grid(N)[:, 1]**2).reshape(N, N)
    out = np.zeros((RO, RO, len(frames)), np.complex64)
    for j, t in enumerate(frames):
        tnv = float(2*tq[t]/Ttime-1); ci = np.zeros((N, N, C), np.complex64)
        for c in range(C):
            v = np.zeros(grid.shape[0], np.complex64)
            for i in range(0, grid.shape[0], chunk):
                cc = grid[i:i+chunk]; tt = torch.full((cc.shape[0],), tnv, device=dev)
                pr = nz.denormalize(cc, m(cc, tt, torch.full((cc.shape[0],), c, dtype=torch.long, device=dev))); v[i:i+chunk] = (pr[:, 0]+1j*pr[:, 1]).cpu().numpy()
            ci[:, :, c] = v.reshape(N, N)
        ci[rrg > 1.0] = 0; im = crop_img(ifft2c_mri(ci), RO, RO); out[:, :, j] = np.sum(np.conj(b1)*im, -1)/den
    return out

def score(dyn, frames, name, check=False):
    rec = np.abs(np.stack([rot(dyn[:, :, j]) for j in range(dyn.shape[2])], -1))
    Trf = Tr[:, :, frames]; s = np.sum(rec[body]*Trf[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s
    if check: RA.check_recon(rec, Trf, mask=body, name=name)
    nrmse = float(np.mean([np.sqrt(np.mean((rec[:, :, j][body]-Trf[:, :, j][body])**2))/rv for j in range(len(frames))]))
    pk = float(Tr[body].max()); mse = float(np.mean([((rec[:, :, j][body]-Trf[:, :, j][body])**2).mean() for j in range(len(frames))]))
    psnr = 10*np.log10(pk**2/(mse+1e-20)); mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev); hs, ss = [], []
    for j, t in enumerate(frames):
        vmax = float(np.percentile(Tr[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, j]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        rt = torch.from_numpy(np.clip(Tr[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu())); ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))
    return dict(psnr=psnr, ssim=float(np.mean(ss)), haarpsi=float(np.mean(hs)), nrmse=nrmse)

def row(nm, r): return f"{nm:22s} PSNR {r['psnr']:6.2f}  SSIM {r['ssim']:.4f}  Haar {r['haarpsi']:.4f}  NRMSE {r['nrmse']:.4f}"
print(f"{'config':22s} {'--- 1x single-FOV (current) ---'}   vs   {'--- 2x oversampled ---'}")
for kind, seeds in [("sub16", [(0, 24000), (1, 40000)]), ("free", [(0, 40000), (1, 34000), (2, 36000)])]:
    for s, st in seeds:
        m = load(kind, s, st)
        if kind == "sub16":
            d1 = render_sub(m, 1)[:, :, FR]; d2 = render_sub(m, 2)[:, :, FR]
        else:
            d1 = render_free(m, 1, FR); d2 = render_free(m, 2, FR)
        r1 = score(d1, FR, f"{kind}_s{s} 1x", check=(s == seeds[0][0])); r2 = score(d2, FR, f"{kind}_s{s} 2x")
        print(row(f"{kind}_s{s} 1x", r1)); print(row(f"{kind}_s{s} 2x", r2),
              f"| dPSNR {r2['psnr']-r1['psnr']:+.2f} dHaar {r2['haarpsi']-r1['haarpsi']:+.4f} dNRMSE {r2['nrmse']-r1['nrmse']:+.4f}", flush=True)
print("DONE_L3")
