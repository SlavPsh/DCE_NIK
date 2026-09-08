"""Real in-vivo test of coil-as-OUTPUT vs coil-as-INPUT, slice 21. Same k-space backbone, data prep
(make_radial_dataset), normalizer, and render (recon_nik_cart) for both; only the coil handling
differs. No ground truth -> compare on HELD-OUT spoke NMSE (generalization) + recon + temporal curves.
Spoke split: train/val(early-stop)/test held out. usage: --coilmode input|output"""
import warnings; warnings.filterwarnings("ignore")
import argparse, time, os, sys, copy, math, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import nik_adapter as A
from nik_model import FourierFeatures, GaborLayer, SineLayer
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
from nik_output_recon import recon_nik_cart
from fftc import ifft2c_mri  # noqa

class GaussianLayer(nn.Module):
    """gaussian-activation inr layer: exp(-0.5*(s*w0*Wx)^2), w0 sets temporal frequency, s the bump width"""
    def __init__(self, in_f, out_f, w0=30.0, s=3.0, is_first=False):
        super().__init__(); self.lin = nn.Linear(in_f, out_f); self.w0 = w0; self.s = s
        with torch.no_grad():
            if is_first: self.lin.weight.uniform_(-1/in_f, 1/in_f)
            else: b = math.sqrt(6/in_f)/w0; self.lin.weight.uniform_(-b, b)
            self.lin.bias.uniform_(-math.pi, math.pi)                              # phase spread so bumps tile time
    def forward(self, x): h = self.w0*self.lin(x); return torch.exp(-0.5*(self.s*h)**2)
_HS = "/home/rnga/vvpshenov/refstage_home"                                # luna-01 cold reads: /home ~2.4x faster than /scratch while FS degraded
REF = _HS if os.path.exists(f"{_HS}/shared.npz") else "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
FIX = dict(depth=12, w0=62.0, s0=15.0, k_freq=256, k_sigma=2.5, t_freq=32, t_sigma=1.5)

class Backbone(nn.Module):
    """shared k-space backbone; coil at input (embedding) or output (heads). basis = learned subspace
    (model=subspace) or FIXED Patlak [AIF, intAIF, 1] + n_free learned atoms (model=f0/f2)."""
    def __init__(self, model_type, ncc, hidden, mode, aif=None, rank=16, phi_w0=30.0, t_sigma=None, act="siren", gauss_s=3.0, t_enc="ff"):
        super().__init__(); self.model_type = model_type; self.C = ncc; self.mode = mode
        self.n_fixed = 0 if model_type == "subspace" else 3
        self.n_free = {"subspace": 0, "f0": 0, "f2": 2}[model_type]
        rank = int(rank) if model_type == "subspace" else {"f0": 3, "f2": 5}[model_type]; self.rank = rank
        self.ff_k = FourierFeatures(2, n_freq=FIX["k_freq"], sigma=FIX["k_sigma"], seed=0)
        self.ff_t = FourierFeatures(1, n_freq=FIX["t_freq"], sigma=(t_sigma if t_sigma else FIX["t_sigma"]), seed=0)  # t_sigma -> temporal FF bandwidth
        cin = 2*FIX["k_freq"] + (8 if mode == "input" else 0)
        if mode == "input": self.coil_embed = nn.Embedding(ncc, 8); nn.init.uniform_(self.coil_embed.weight, -1, 1)
        self.a_first = GaborLayer(cin, hidden, w0=FIX["w0"], s0=FIX["s0"], is_first=True)
        self.a_blocks = nn.ModuleList([GaborLayer(2*hidden, hidden, w0=FIX["w0"], s0=FIX["s0"]) for _ in range(FIX["depth"]-2)])
        self.a_head = nn.Linear(2*hidden, 2*rank*(1 if mode == "input" else ncc))
        n_learn = rank if model_type == "subspace" else self.n_free                # learned temporal atoms
        self.act = act; self.t_enc = t_enc                                         # t_enc: 'ff' fourier-feature time input, 'raw' feeds t directly
        if n_learn > 0:                                                            # temporal net, activation swappable (phi_w0 = temporal freq for all)
            din = 2*FIX["t_freq"] if t_enc == "ff" else 1
            if act == "siren":
                ph = [SineLayer(din, 64, w0=phi_w0, is_first=True)] + [SineLayer(64, 64, w0=phi_w0) for _ in range(2)]
            elif act == "gabor":                                                   # wire; GaborLayer emits 2x (complex reim), so out=32 -> 64
                ph = [GaborLayer(din, 32, w0=phi_w0, s0=FIX["s0"], is_first=True)] + [GaborLayer(64, 32, w0=phi_w0, s0=FIX["s0"]) for _ in range(2)]
            elif act == "gaussian":                                                # gauss_s = bump width (sharpness); s*w0 sets effective sharpness
                ph = [GaussianLayer(din, 64, w0=phi_w0, s=gauss_s, is_first=True)] + [GaussianLayer(64, 64, w0=phi_w0, s=gauss_s) for _ in range(2)]
            else: raise ValueError(act)
            self.phi_body = nn.Sequential(*ph); self.phi_head = nn.Linear(64, 2*n_learn)
        if aif is not None:
            self.register_buffer("aif_tgrid", torch.as_tensor(aif["tgrid"], dtype=torch.float32))
            self.register_buffer("aif_vals", torch.as_tensor(aif["aif"], dtype=torch.float32))
            self.register_buffer("iaif_vals", torch.as_tensor(aif["iaif"], dtype=torch.float32))
    def amp_in(self, k, coil):
        h = self.a_first(torch.cat([self.ff_k(k), self.coil_embed(coil)], -1))
        for blk in self.a_blocks: h = h + blk(h)
        a = self.a_head(h).view(-1, self.rank, 2); return a[..., 0]+1j*a[..., 1]
    def amp_out(self, k):
        h = self.a_first(self.ff_k(k))
        for blk in self.a_blocks: h = h + blk(h)
        a = self.a_head(h).view(-1, self.rank, self.C, 2); return a[..., 0]+1j*a[..., 1]
    def _interp(self, t, vals):
        tg = self.aif_tgrid; t = t.clamp(float(tg[0]), float(tg[-1])); i = torch.searchsorted(tg, t).clamp(1, tg.numel()-1)
        t0, t1 = tg[i-1], tg[i]; w = (t-t0)/(t1-t0+1e-12); return vals[i-1]*(1-w) + vals[i]*w
    def basis(self, t):                                                            # [N,rank] complex
        t = t.view(-1)
        tin = self.ff_t(t.view(-1, 1)) if self.t_enc == "ff" else t.view(-1, 1)    # ff vs raw time input
        if self.model_type == "subspace":
            h = self.phi_head(self.phi_body(tin)).view(-1, self.rank, 2); return h[..., 0]+1j*h[..., 1]
        real = torch.stack([self._interp(t, self.aif_vals), self._interp(t, self.iaif_vals), torch.ones_like(t)], 1)  # [N,3] fixed
        P = torch.zeros(t.numel(), self.rank, 2, device=t.device); P[:, :self.n_fixed, 0] = real
        if self.n_free > 0:
            P[:, self.n_fixed:, :] = self.phi_head(self.phi_body(tin)).view(-1, self.n_free, 2)
        return P[..., 0]+1j*P[..., 1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--coilmode", choices=["input", "output"], required=True)
    ap.add_argument("--model", choices=["subspace", "f0", "f2"], default="subspace")
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=768); ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--slice", type=int, default=21)
    ap.add_argument("--phi_w0", type=float, default=30.0); ap.add_argument("--t_sigma", type=float, default=0.0)
    ap.add_argument("--act", choices=["siren", "gabor", "gaussian"], default="siren")
    ap.add_argument("--gauss_s", type=float, default=3.0); ap.add_argument("--t_enc", choices=["ff", "raw"], default="ff")
    ap.add_argument("--tag", default=""); a = ap.parse_args()
    dev = torch.device("cuda"); sh = A.load_shared(REF); ds = A.make_radial_dataset(REF, a.slice, shared=sh)
    m = ds["meta"]; nx, ntv, C, nt, bas = m["nx"], m["ntviews"], m["ncc"], m["nt"], m["bas"]; M = nx*ntv
    b1 = ds["b1"].astype(np.complex64)
    coords = ds["x_all"][:M]; times = ds["t_all"][:M]; spoke = ds["spoke_id_all"][:M].cpu().numpy()   # per-(k,t)
    Yc = ds["y_all_raw"].view(C, M, 2)[..., 0] + 1j*ds["y_all_raw"].view(C, M, 2)[..., 1]              # [C,M]
    Yc = Yc.T.contiguous()                                                                              # [M,C] complex raw
    # spoke split (held out for val/test): distribute across acquisition
    sp = spoke % 10; train = torch.tensor(sp < 8, device=dev); val = torch.tensor(sp == 8, device=dev); test = torch.tensor(sp == 9, device=dev)
    print(f"real slice{a.slice} [{a.coilmode}]: M={M} nx={nx} ntv={ntv} C={C} nt={nt} | train {int(train.sum())} val {int(val.sum())} test {int(test.sum())}", flush=True)
    # normalizer: fit on TRAIN samples (all coils), envelope 0.75, no dcf (match real pipeline dcf_power 0)
    xt = coords[train]; yt = ds["y_all_raw"].view(C, M, 2)                                             # for fit use coil-flattened train
    xfit = xt.repeat(C, 1); yfit = torch.cat([yt[c][train] for c in range(C)], 0)
    dcf = torch.ones(xfit.shape[0], device=dev); nz = KSpaceNormalizer(); nz.fit(xfit, yfit, dcf=dcf, envelope_exponent=0.75)
    def norm(x, Yreal):                                                                                 # normalize [Npts,C] complex
        out = torch.empty_like(Yreal)
        for c in range(C): out[:, c] = torch.view_as_complex(nz.normalize(x, torch.view_as_real(Yreal[:, c].contiguous())).contiguous())
        return out
    Yn = norm(coords, Yc)                                                                               # [M,C] normalized
    aifd = None
    if a.model in ("f0", "f2"):
        az = np.load(f"/scratch/rnga/vvpshenov/DCE_NIK/aif_slice{a.slice}.npz"); TA = float(az["tC"][-1])
        aif = az["aif_frame"].astype(np.float64); aif = aif/(aif.max()+1e-9)
        iaif = np.concatenate([[0.0], np.cumsum(0.5*(aif[1:]+aif[:-1])*np.diff(az["tC"]))]); iaif = iaif/(iaif.max()+1e-9)
        aifd = dict(tgrid=2.0*(az["tC"]/TA)-1.0, aif=aif, iaif=iaif)
    model = Backbone(a.model, C, a.hidden, a.coilmode, aif=aifd, rank=a.rank, phi_w0=a.phi_w0, t_sigma=(a.t_sigma or None), act=a.act, gauss_s=a.gauss_s, t_enc=a.t_enc).to(dev); R = model.rank
    print(f"model={a.model} coil={a.coilmode} rank={R} phi_w0={a.phi_w0} t_sigma={a.t_sigma} act={a.act} gauss_s={a.gauss_s} t_enc={a.t_enc} n_free={model.n_free}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=3e-3)
    tr_idx = torch.where(train)[0]; ntr = tr_idx.numel(); rng = np.random.RandomState(0)
    ci_tile = torch.arange(C, device=dev)                                                              # for vectorized input forward
    def predict(x, t):                                                                                  # [B,C] complex
        phi = model.basis(t)
        if a.coilmode == "output": return torch.einsum("brc,br->bc", model.amp_out(x), phi)
        B = x.shape[0]; amp = model.amp_in(x.repeat(C, 1), ci_tile.repeat_interleave(B)).view(C, B, model.rank)  # one batched pass
        return torch.einsum("cbr,br->bc", amp, phi)
    @torch.no_grad()
    def heldout_nmse(mask):
        model.eval(); idx = torch.where(mask)[0]; num = dn = 0.0
        for i in range(0, idx.numel(), 20000):
            j = idx[i:i+20000]; pr = predict(coords[j], times[j])
            num += float((torch.abs(pr-Yn[j])**2).sum()); dn += float((torch.abs(Yn[j])**2).sum())
        model.train(); return num/(dn+1e-30)
    t0 = time.time(); best = (1e30, 0, None)
    for step in range(1, a.steps+1):
        j = tr_idx[torch.tensor(rng.choice(ntr, a.batch, replace=False), device=dev)]
        loss = torch.mean(torch.abs(predict(coords[j], times[j]) - Yn[j])**2)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % (a.steps//30) == 0 or step == 1:
            v = heldout_nmse(val)
            if v < best[0]: best = (v, step, copy.deepcopy({k: t.detach().cpu() for k, t in model.state_dict().items()}))
            print(f"  step {step:6d} loss {float(loss):.4e} val {v:.4e} best@{best[1]} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict({k: t.to(dev) for k, t in best[2].items()}); model.eval()
    test_nmse = heldout_nmse(test); print(f"  VAL-SELECTED best step {best[1]} val {best[0]:.4e} | TEST held-out NMSE {test_nmse:.4e}", flush=True)
    # render on GPU: numpy ifft2c over [nx,nx,nt,C] was ~9min on CPU while the A100 sat idle
    rt = time.time()
    NG = nx; grid = torch.from_numpy(A.cartesian_grid(NG)).to(dev)
    rrg = torch.from_numpy(np.sqrt(A.cartesian_grid(NG)[:, 0]**2+A.cartesian_grid(NG)[:, 1]**2).reshape(NG, NG)).to(dev)
    ftn = torch.tensor(2.0*sh["frame_time"]-1.0, dtype=torch.float32, device=dev)
    b1t = torch.from_numpy(b1).to(dev)                                                                   # [nx,nx,C] complex
    den = (b1t.abs()**2).sum(2)[:, :, None] + 1e-12                                                      # [nx,nx,1]
    x0 = (NG - bas)//2                                                                                   # central crop start (crop_img)
    def ifft2c_g(X):                                                                                     # matches fftc.ifft2c_mri: fftshift(fft(fftshift))/sqrt(N) per axis
        x = torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(X, dim=0), dim=0), dim=0) / (X.shape[0]**0.5)
        return torch.fft.fftshift(torch.fft.fft(torch.fft.fftshift(x, dim=1), dim=1), dim=1) / (X.shape[1]**0.5)
    with torch.no_grad():
        Phi = model.basis(ftn).to(torch.complex64)                                                      # [nt,R]
        coeff = torch.zeros((NG, NG, R, C), dtype=torch.complex64, device=dev)
        for i in range(0, grid.shape[0], 200000):
            gg = grid[i:i+200000]; B = gg.shape[0]
            amp = model.amp_out(gg) if a.coilmode == "output" else model.amp_in(gg.repeat(C, 1), ci_tile.repeat_interleave(B)).view(C, B, R).permute(1, 2, 0)  # [B,R,C]
            coeff.view(-1, R, C)[i:i+B] = amp * nz.get_pointwise_scale(gg).view(B, 1, 1)                 # denorm all R,C at once
        coeff[rrg > 1.0] = 0
        outs = []                                                                                        # chunk over frames (GPU mem)
        for t0 in range(0, nt, 32):
            cart = torch.einsum("xyrc,tr->xytc", coeff, Phi[t0:t0+32])                                   # [nx,nx,chunk,C]
            comb = (ifft2c_g(cart) * torch.conj(b1t)[:, :, None, :]).sum(3) / den                        # [nx,nx,chunk]
            g = comb[x0:x0+bas, x0:x0+bas].abs().float().cpu().numpy()                                   # crop + magnitude
            if t0 == 0 and os.environ.get("CHECK_RENDER"):                                               # one-chunk parity vs CPU recon_nik_cart
                c = recon_nik_cart(cart.cpu().numpy().astype(np.complex64), b1, bas)
                print(f"  RENDER-CHECK max|gpu-cpu|={np.abs(g-c).max():.3e} rel={np.abs(g-c).max()/(c.max()+1e-9):.2e}", flush=True)
            outs.append(g)
        img = np.concatenate(outs, -1)                                                                   # [bas,bas,nt]
    print(f"  render {img.shape} in {time.time()-rt:.0f}s (GPU)", flush=True)
    np.save(f"/scratch/rnga/vvpshenov/DCE_NIK/results/realdata_nik_vs_cs_figures/outcoil_{a.model}_{a.coilmode}{a.tag}_slice{a.slice}.npy", img.astype(np.float32))
    print(f"REAL-{a.model}-{a.coilmode.upper()} slice{a.slice}: TEST held-out NMSE {test_nmse:.4e} | recon {img.shape} saved", flush=True)
    print("DONE_OCREAL", flush=True)

if __name__ == "__main__": main()
