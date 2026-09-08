"""Decider for K3a: does the RENDERED recon still satisfy low-|k| data consistency, or does the
rank-R + gridding + SENSE render lose it? Forward-project the SENSE render to the measured spokes
(nufft2d2) and compare to measured y at |k|<0.10, vs the raw-network radial misfit (K1a). If the
render misfit is also ~40 dB, a DC step has no headroom (K3a null is real). If larger, a proper
CG-DC projection is warranted."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob, finufft
import xph_pipeline as P, xph_common as X
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
d = P.data(); tq = d["times"]; kx = d["kx"]; ky = d["ky"]; b1 = d["b1"].astype(np.complex128); RO = b1.shape[0]; F = len(tq)
tr = np.array(P.TRAIN_ANG); Ttime = float(tq.max())

def build(kind, s, st):
    tag = f"{'sub16' if kind=='sub16' else 'free'}_w768_s{s}"; p = f"{P.OUT}/checkpoints/{tag}/ck_{st}.pt"
    if not os.path.exists(p): p = sorted(glob.glob(f"{P.OUT}/checkpoints/{tag}/ck_*.pt"))[-1]
    m = P.make_model_g("wire_ff_subspace" if kind=="sub16" else "wire_ff", 768, P.FIX["k_sigma"], 0, C, dev, rank=16, warmstart=False)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); m.eval(); return m

@torch.no_grad()
def render_misfit(m):
    dyn = P.reconstruct_g(m, nz, tq, dev)                                          # native coil-combined [RO,RO,F]
    num_r = den_r = num_n = den_n = 0.0
    for t in range(F):
        fx = np.ascontiguousarray((2*np.pi*kx[t, tr]).reshape(-1).astype(np.float64)); fy = np.ascontiguousarray((2*np.pi*ky[t, tr]).reshape(-1).astype(np.float64))
        rr = (np.abs(kx[t, tr]+1j*ky[t, tr]).reshape(-1)/0.5); low = rr < 0.10
        # RENDER forward: per-coil SENSE forward of the rendered image, type-2 NUFFT
        cimg = np.ascontiguousarray((dyn[:, :, t][None]*np.transpose(b1, (2, 0, 1))).astype(np.complex128))  # [C,RO,RO]
        pred = finufft.nufft2d2(fx, fy, cimg, isign=-1, eps=1e-6)                   # [C,M]
        meas = d["kdata"][:, t, tr, :].reshape(C, -1).astype(np.complex128)
        # scale-match (global complex) to remove render normalization, then low-|k| relative misfit
        sc = np.sum(np.conj(pred)*meas)/(np.sum(np.abs(pred)**2)+1e-30); pred *= sc
        num_r += (np.abs(pred[:, low]-meas[:, low])**2).sum(); den_r += (np.abs(meas[:, low])**2).sum()
        # raw NETWORK radial misfit at same low-|k| points (K1a-style)
        cx = torch.tensor(np.stack([2*kx[t, tr].reshape(-1), 2*ky[t, tr].reshape(-1)], 1), dtype=torch.float32, device=dev)
        tt = torch.full((cx.shape[0],), 2*tq[t]/Ttime-1, dtype=torch.float32, device=dev)
        for c in range(C):
            pr = nz.denormalize(cx, m(cx, tt, torch.full((cx.shape[0],), c, dtype=torch.long, device=dev)))
            yh = (pr[:, 0]+1j*pr[:, 1]).cpu().numpy(); yt = d["kdata"][c, t, tr, :].reshape(-1)
            num_n += (np.abs(yh[low]-yt[low])**2).sum(); den_n += (np.abs(yt[low])**2).sum()
    return num_r/den_r, num_n/den_n

for kind, s, st in [("sub16", 0, 24000), ("free", 0, 40000)]:
    mr, mn = render_misfit(build(kind, s, st))
    print(f"{kind}_s{s}: RENDER low-|k| misfit NMSE {mr:.3e} ({10*np.log10(1/mr):.1f} dB) | raw-NETWORK {mn:.3e} ({10*np.log10(1/mn):.1f} dB)", flush=True)
print("DONE_RMF")
