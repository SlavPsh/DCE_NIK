"""Coil-as-OUTPUT NIK (user's test): instead of f(k,t,coil_embedding)->1 value (sub16, coil in the
input), g(k,t)->C values (one head per coil), each matched to that coil's measured k-space. All coils
SHARE one k-space backbone (FF+Gabor, same freq params as sub16) and differ only at the output head.
Everything else -- k-space representation, temporal subspace, normalization, 2x render, scoring -- is
IDENTICAL to sub16, so this isolates the coil parameterization. Compare vs sub16 (2x) and truth."""
import warnings; warnings.filterwarnings("ignore")
import argparse, time, os, numpy as np, torch, torch.nn as nn
import xph_pipeline as P, xph_common as X, recon_asserts as RA, nik_adapter as A
from nik_model import FourierFeatures, GaborLayer, SineLayer
from fftc import ifft2c_mri, crop_img
from masked_metrics import haarpsi_masked, ssim_masked

class OutCoil(nn.Module):
    """sub16's amplitudes backbone with coil moved to the output head (R*C), shared temporal basis."""
    def __init__(self, ncc, rank, hidden, depth, k_freq, k_sigma, w0, s0, t_freq, t_sigma, ff_seed):
        super().__init__(); self.rank = rank; self.C = ncc
        self.ff_k = FourierFeatures(2, n_freq=k_freq, sigma=k_sigma, seed=ff_seed)
        self.ff_t = FourierFeatures(1, n_freq=t_freq, sigma=t_sigma, seed=ff_seed)
        self.a_first = GaborLayer(2*k_freq, hidden, w0=w0, s0=s0, is_first=True)
        self.a_blocks = nn.ModuleList([GaborLayer(2*hidden, hidden, w0=w0, s0=s0) for _ in range(max(0, depth-2))])
        self.a_head = nn.Linear(2*hidden, 2*rank*ncc)                              # <-- coil at OUTPUT
        ph = [SineLayer(2*t_freq, 64, w0=30.0, is_first=True)] + [SineLayer(64, 64, w0=30.0) for _ in range(2)]
        self.phi_body = nn.Sequential(*ph); self.phi_head = nn.Linear(64, 2*rank)
    def amplitudes(self, k):                                                       # [N,2] -> [N,R,C] complex
        h = self.a_first(self.ff_k(k))
        for blk in self.a_blocks: h = h + blk(h)
        a = self.a_head(h).view(-1, self.rank, self.C, 2); return a[..., 0] + 1j*a[..., 1]
    def basis(self, t):                                                            # [N] -> [N,R] complex
        h = self.phi_head(self.phi_body(self.ff_t(t.view(-1, 1)))).view(-1, self.rank, 2); return h[..., 0] + 1j*h[..., 1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--batch", type=int, default=16384); ap.add_argument("--lr", type=float, default=1e-5); a = ap.parse_args()
    dev = torch.device("cuda"); FIX = P.FIX
    X_, Yn, T_, C_, nz, dims = P.build_train(dev); C = dims[3]                      # nz = same normalizer as sub16
    d = P.data(); tq = d["times"]; F = len(tq); RO = d["b1"].shape[0]; tr = np.array(P.TRAIN_ANG); Ttime = float(tq.max())
    body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"]); rv = float(Tr[body].max()-Tr[body].min())
    b1 = d["b1"].astype(np.complex64)
    # per-(k,t) train points, coords=2k (sub16 convention), all coils normalized with nz
    coords, times, Yc = [], [], []
    for t in range(F):
        kx = d["kx"][t, tr].reshape(-1); ky = d["ky"][t, tr].reshape(-1); M = kx.size
        coords.append(np.stack([2*kx, 2*ky], 1)); times.append(np.full(M, 2*tq[t]/Ttime-1, np.float32))
        Yc.append(d["kdata"][:, t, tr, :].reshape(C, M).T)                          # [M,C]
    coords = torch.tensor(np.concatenate(coords), dtype=torch.float32, device=dev); times = torch.tensor(np.concatenate(times), dtype=torch.float32, device=dev)
    Yc = np.concatenate(Yc)                                                         # [N,C] complex (raw)
    Yn_c = np.stack([nz.normalize(coords, torch.tensor(np.stack([Yc[:, c].real, Yc[:, c].imag], 1), dtype=torch.float32, device=dev)).cpu().numpy() for c in range(C)], 1)  # [N,C,2]
    Yn_c = torch.tensor(Yn_c[:, :, 0] + 1j*Yn_c[:, :, 1], device=dev)               # [N,C] normalized complex
    N = coords.shape[0]; print(f"outcoil: N={N} points, C={C}, rank={a.rank}", flush=True)
    # VAL points (held-out angle 5, same as sub16) for early stopping
    va = np.array(P.VAL_ANG); vc, vt, vY = [], [], []
    for t in range(F):
        kx = d["kx"][t, va].reshape(-1); ky = d["ky"][t, va].reshape(-1); M = kx.size
        vc.append(np.stack([2*kx, 2*ky], 1)); vt.append(np.full(M, 2*tq[t]/Ttime-1, np.float32)); vY.append(d["kdata"][:, t, va, :].reshape(C, M).T)
    vcoords = torch.tensor(np.concatenate(vc), dtype=torch.float32, device=dev); vtimes = torch.tensor(np.concatenate(vt), dtype=torch.float32, device=dev)
    vYc = np.concatenate(vY); vYn = np.stack([nz.normalize(vcoords, torch.tensor(np.stack([vYc[:, c].real, vYc[:, c].imag], 1), dtype=torch.float32, device=dev)).cpu().numpy() for c in range(C)], 1)
    vYn = torch.tensor(vYn[:, :, 0] + 1j*vYn[:, :, 1], device=dev)
    @torch.no_grad()
    def val_nmse():
        model.eval(); num = 0.0; den = 0.0
        for i in range(0, vcoords.shape[0], 40000):
            pr = torch.einsum("brc,br->bc", model.amplitudes(vcoords[i:i+40000]), model.basis(vtimes[i:i+40000]))
            num += float((torch.abs(pr - vYn[i:i+40000])**2).sum()); den += float((torch.abs(vYn[i:i+40000])**2).sum())
        model.train(); return num/(den+1e-30)
    model = OutCoil(C, a.rank, a.width, FIX["depth"], FIX["k_freq"], FIX["k_sigma"], FIX["w0"], FIX["s0"], FIX["t_freq"], FIX["t_sigma"], a.seed).to(dev); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=P.WD); t0 = time.time(); rng = np.random.RandomState(a.seed)
    import copy; best = (1e30, 0, None)                                             # (val_nmse, step, state)
    for step in range(1, a.steps+1):
        idx = torch.tensor(rng.choice(N, a.batch, replace=False), device=dev)
        amp = model.amplitudes(coords[idx]); phi = model.basis(times[idx])          # [B,R,C],[B,R]
        pred = torch.einsum("brc,br->bc", amp, phi)                                 # [B,C]
        loss = torch.mean(torch.abs(pred - Yn_c[idx])**2)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % (a.steps//40) == 0 or step == 1:
            v = val_nmse()
            if v < best[0]: best = (v, step, copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()}))
            print(f"  step {step:6d} loss {float(loss):.4e} val {v:.4e} best@{best[1]} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict({k: t.to(dev) for k, t in best[2].items()})               # <-- val-selected best checkpoint
    print(f"  VAL-SELECTED best step {best[1]} (val {best[0]:.4e})", flush=True)
    # ---- render: 2x oversample, denorm, ifft per coil, SENSE combine, dyn = thC @ basis(t) (== sub16 render) ----
    model.eval(); NG = 2*RO
    grid = torch.from_numpy(A.cartesian_grid(NG)).to(dev); rrg = np.sqrt(A.cartesian_grid(NG)[:, 0]**2 + A.cartesian_grid(NG)[:, 1]**2).reshape(NG, NG)
    with torch.no_grad():
        amp = model.amplitudes(grid)                                               # [Pg,R,C] complex
        cart = np.zeros((NG, NG, a.rank, C), np.complex64)
        for r in range(a.rank):
            for c in range(C):
                pr = nz.denormalize(grid, torch.view_as_real(amp[:, r, c].contiguous())); cart[:, :, r, c] = (pr[:, 0]+1j*pr[:, 1]).cpu().numpy().reshape(NG, NG)
        cart[rrg > 1.0] = 0
        a_rc = crop_img(ifft2c_mri(cart.reshape(NG, NG, a.rank*C)).reshape(NG, NG, a.rank, C), RO, RO)
        thC = np.stack([np.sum(np.conj(b1)*a_rc[:, :, r, :], -1)/(np.sum(np.abs(b1)**2, -1)+1e-8) for r in range(a.rank)], -1)  # SENSE combine
        tn = torch.tensor([2*tq[t]/Ttime-1 for t in range(F)], dtype=torch.float32, device=dev); Phi = model.basis(tn).cpu().numpy()  # [F,R] complex
        dyn = np.einsum("xyr,tr->xyt", thC, Phi).astype(np.complex64)
    rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1))
    rec = np.abs(np.stack([rot(dyn[:, :, t]) for t in range(F)], -1))
    s = np.sum(rec[body]*Tr[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s
    try: RA.check_recon(rec, Tr, mask=body, name="outcoil"); print("  check_recon PASS", flush=True)
    except Exception as e: print(f"  check_recon NOTE: {e}", flush=True)
    nrmse = float(np.mean([np.sqrt(np.mean((rec[:, :, t][body]-Tr[:, :, t][body])**2))/rv for t in range(F)]))
    pk = float(Tr[body].max()); mse = float(np.mean([((rec[:, :, t][body]-Tr[:, :, t][body])**2).mean() for t in range(0, F, 2)]))
    psnr = 10*np.log10(pk**2/(mse+1e-20)); mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev); hs, ss = [], []
    for t in range(0, F, 2):
        vmax = float(np.percentile(Tr[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev); rt = torch.from_numpy(np.clip(Tr[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu())); ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))
    cur = {nm: float(np.linalg.norm(rec[Rz[nm]].mean(0)-Tr[Rz[nm]].mean(0))/(np.linalg.norm(Tr[Rz[nm]].mean(0))+1e-12)) for nm in ("aorta", "cortex", "medulla")}
    np.savez(f"{P.OUT}/arrays/outcoil_r{a.rank}_s{a.seed}.npz", rec=rec.astype(np.float32))
    print(f"\nOUTCOIL rank{a.rank} seed{a.seed}: PSNR {psnr:.2f} SSIM {np.mean(ss):.4f} HaarPSI {np.mean(hs):.4f} NRMSE {nrmse:.4f} | aortaC {cur['aorta']:.3f} cortexC {cur['cortex']:.3f} medullaC {cur['medulla']:.3f}", flush=True)
    print("baseline sub16(2x):   PSNR 36.12 SSIM 0.9218 HaarPSI 0.8856 NRMSE 0.0153 | aortaC 0.120 cortexC 0.055 medullaC 0.052 (input-coil)", flush=True)
    print("DONE_OUTCOIL", flush=True)

if __name__ == "__main__": main()
