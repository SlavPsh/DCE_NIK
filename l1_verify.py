"""L1: verify the network(39.5dB)->render(22.6dB) low-|k| consistency loss is REAL and like-for-like.
Both quantities measured at the SAME train-spoke coordinates, TRUTH frame, denormalized (physical)
scale, NO per-frame scale-match on either side (the network is already at the measured scale via
denorm; the render inherits that scale through ifft+SENSE, so a fair comparison must not re-scale).
  p_net = denorm(network(spoke coords))              [the 39.5 dB reference; direct k-space]
  p_ren = nufft2d2(b1 * rot(dyn))  at spoke coords    [render forward-projection]
Report vs measured (net, ren) AND render-vs-network directly (isolates the render round trip, no
truth/measured). If p_ren ~ p_net -> no render loss, lead dead. Also print WITH scale-match to expose
a pure scale/phase offset vs a structural loss. sub16 s0. No training; truth not used."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob, finufft
import xph_pipeline as P, xph_common as X
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
d = P.data(); tq = d["times"]; kx = d["kx"]; ky = d["ky"]; b1 = d["b1"].astype(np.complex128); RO = b1.shape[0]; F = len(tq)
tr = np.array(P.TRAIN_ANG); b1c = np.ascontiguousarray(np.transpose(b1, (2, 0, 1))); Ttime = float(tq.max())
rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1))

tag = "sub16_w768_s0"; p = f"{P.OUT}/checkpoints/{tag}/ck_24000.pt"
if not os.path.exists(p): p = sorted(glob.glob(f"{P.OUT}/checkpoints/{tag}/ck_*.pt"))[-1]
m = P.make_model_g("wire_ff_subspace", 768, P.FIX["k_sigma"], 0, C, dev, rank=16, warmstart=False)
m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); m.eval()
with torch.no_grad(): dyn = P.reconstruct_g(m, nz, tq, dev)
dyn_tf = np.stack([rot(dyn[:, :, t]) for t in range(F)], -1)                       # truth frame

def db(num, den): return 10*np.log10((den+1e-30)/(num+1e-30))
acc = {k: [0.0, 0.0] for k in ["net", "ren", "ren_sc", "rvn", "rvn_sc"]}           # [num, den]
for band, (lo, hi) in [("low(<0.10)", (0, 0.10)), ("all", (0, 1.01))]:
    tot = {k: [0.0, 0.0] for k in acc}
    for t in range(F):
        rr = np.abs(kx[t, tr]+1j*ky[t, tr]).reshape(-1)/0.5; sel = (rr >= lo) & (rr < hi)
        if sel.sum() == 0: continue
        fx = np.ascontiguousarray((2*np.pi*kx[t, tr]).reshape(-1).astype(np.float64)); fy = np.ascontiguousarray((2*np.pi*ky[t, tr]).reshape(-1).astype(np.float64))
        meas = d["kdata"][:, t, tr, :].reshape(C, -1).astype(np.complex128)         # [C,M]
        # network prediction at the SAME spoke coords (denormalized, per coil)
        cx = torch.tensor(np.stack([2*kx[t, tr].reshape(-1), 2*ky[t, tr].reshape(-1)], 1), dtype=torch.float32, device=dev)
        tt = torch.full((cx.shape[0],), 2*tq[t]/Ttime-1, dtype=torch.float32, device=dev)
        pnet = np.empty((C, cx.shape[0]), np.complex128)
        with torch.no_grad():
            for c in range(C):
                pr = nz.denormalize(cx, m(cx, tt, torch.full((cx.shape[0],), c, dtype=torch.long, device=dev)))
                pnet[c] = (pr[:, 0]+1j*pr[:, 1]).cpu().numpy()
        pren = finufft.nufft2d2(fx, fy, np.ascontiguousarray(dyn_tf[:, :, t][None]*b1c), isign=-1, eps=1e-6)  # [C,M]
        M = meas[:, sel]; N = pnet[:, sel]; Rn = pren[:, sel]
        sc = np.sum(np.conj(Rn)*M)/(np.sum(np.abs(Rn)**2)+1e-30)                     # per-frame complex scale (render->meas)
        scn = np.sum(np.conj(Rn)*N)/(np.sum(np.abs(Rn)**2)+1e-30)                    # render->network
        tot["net"][0] += (np.abs(N-M)**2).sum(); tot["net"][1] += (np.abs(M)**2).sum()
        tot["ren"][0] += (np.abs(Rn-M)**2).sum(); tot["ren"][1] += (np.abs(M)**2).sum()
        tot["ren_sc"][0] += (np.abs(sc*Rn-M)**2).sum(); tot["ren_sc"][1] += (np.abs(M)**2).sum()
        tot["rvn"][0] += (np.abs(Rn-N)**2).sum(); tot["rvn"][1] += (np.abs(N)**2).sum()
        tot["rvn_sc"][0] += (np.abs(scn*Rn-N)**2).sum(); tot["rvn_sc"][1] += (np.abs(N)**2).sum()
    print(f"\n=== band {band} ===")
    print(f"  network vs measured (denorm, no scale)   : {db(*tot['net']):6.2f} dB   [reference]")
    print(f"  render  vs measured (no scale)           : {db(*tot['ren']):6.2f} dB")
    print(f"  render  vs measured (per-frame scale)    : {db(*tot['ren_sc']):6.2f} dB")
    print(f"  render  vs NETWORK  (round-trip, no scale): {db(*tot['rvn']):6.2f} dB   [isolates render, no meas/truth]")
    print(f"  render  vs NETWORK  (round-trip, w/ scale): {db(*tot['rvn_sc']):6.2f} dB")
print("\nDONE_L1")
