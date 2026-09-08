"""L2a: localize the ~17 dB render-vs-network loss by inserting the render pipeline stage by stage
and forward-projecting each intermediate back to the SAME train-spoke coords, per |k| band, vs the
raw per-coil network prediction (reference). Same per-frame complex scale + rot(truth-frame) forward
convention throughout, so the L1 units mismatch cannot reappear and only stage-to-stage drops localize
the loss. crop_img(.,RO,RO) is a no-op (verified) so it is not a separate stage.
  ref  : amp(spoke)@basis            (per coil, denormalized) = network k-space at spokes
  S1   : grid query round-trip       amp(GRID)@basis -> ifft2c -> forward-nufft(spoke)   [c+e+g]
  S2   : + |k|>1 zeroing             (d)
  S3   : + SENSE combine/re-expand   full render dyn, b1*dyn -> forward-nufft              [f]
Also the SENSE round-trip in isolation (image domain). sub16 s0, frames subsampled. No training."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob, finufft
import xph_pipeline as P, xph_common as X, nik_adapter as A
from fftc import ifft2c_mri
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
d = P.data(); tq = d["times"]; kx = d["kx"]; ky = d["ky"]; b1 = d["b1"].astype(np.complex128); RO = b1.shape[0]; F = len(tq)
tr = np.array(P.TRAIN_ANG); b1c = np.ascontiguousarray(np.transpose(b1, (2, 0, 1))); Ttime = float(tq.max())
rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1)); den = np.sum(np.abs(b1)**2, -1)+1e-8
FR = np.arange(0, F, 4)                                                            # subsample frames

tag = "sub16_w768_s0"; p = f"{P.OUT}/checkpoints/{tag}/ck_24000.pt"
if not os.path.exists(p): p = sorted(glob.glob(f"{P.OUT}/checkpoints/{tag}/ck_*.pt"))[-1]
m = P.make_model_g("wire_ff_subspace", 768, P.FIX["k_sigma"], 0, C, dev, rank=16, warmstart=False)
m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); m.eval()

# temporal basis + grid coefficient k-space (cart) exactly as extract_coeffs_g builds it
tn_all = torch.tensor([2*tq[t]/Ttime-1 for t in range(F)], dtype=torch.float32, device=dev)
with torch.no_grad(): Phi = m.basis(tn_all).cpu().numpy()[:, :, 0].astype(np.complex128)   # [T,Rr]
Rr = Phi.shape[1]
grid = torch.from_numpy(A.cartesian_grid(RO)).to(dev); rrg = np.sqrt(A.cartesian_grid(RO)[:, 0]**2+A.cartesian_grid(RO)[:, 1]**2).reshape(RO, RO)
cart = np.zeros((RO, RO, Rr, C), np.complex128)
with torch.no_grad():
    for c in range(C):
        Amp = m.amplitudes(grid, torch.full((grid.shape[0],), c, dtype=torch.long, device=dev))   # [P,Rr,2]
        out = np.zeros((grid.shape[0], Rr), np.complex128)
        for r in range(Rr):
            pr = nz.denormalize(grid, Amp[:, r, :].contiguous()); out[:, r] = (pr[:, 0]+1j*pr[:, 1]).cpu().numpy()
        cart[:, :, :, c] = out.reshape(RO, RO, Rr)
# full render coeff maps (SENSE-combined) via the real function -> guarantees S3 == actual render
thC = P.extract_coeffs_g(m, nz, dev)                                              # [RO,RO,Rr]

def bands(rr):
    return {"low(<0.10)": rr < 0.10, "mid(0.10-0.5)": (rr >= 0.10) & (rr < 0.5), "high(>0.5)": rr >= 0.5}
acc = {s: {b: [0.0, 0.0] for b in ["low(<0.10)", "mid(0.10-0.5)", "high(>0.5)", "all"]} for s in ["S1", "S2", "S3"]}
sense_num = sense_den = 0.0
cart_z = cart.copy(); cart_z[rrg > 1.0] = 0                                       # |k|>1 zeroed copy

for t in FR:
    fx = np.ascontiguousarray((2*np.pi*kx[t, tr]).reshape(-1).astype(np.float64)); fy = np.ascontiguousarray((2*np.pi*ky[t, tr]).reshape(-1).astype(np.float64))
    rr = np.abs(kx[t, tr]+1j*ky[t, tr]).reshape(-1)/0.5; bb = bands(rr)
    # reference: per-coil network k-space at spokes
    cx = torch.tensor(np.stack([2*kx[t, tr].reshape(-1), 2*ky[t, tr].reshape(-1)], 1), dtype=torch.float32, device=dev)
    tt = torch.full((cx.shape[0],), 2*tq[t]/Ttime-1, dtype=torch.float32, device=dev)
    ref = np.empty((C, cx.shape[0]), np.complex128)
    with torch.no_grad():
        for c in range(C):
            prr = nz.denormalize(cx, m(cx, tt, torch.full((cx.shape[0],), c, dtype=torch.long, device=dev))); ref[c] = (prr[:, 0]+1j*prr[:, 1]).cpu().numpy()
    # S1: grid query, per coil, no zero
    s1 = np.stack([finufft.nufft2d2(fx, fy, np.ascontiguousarray(rot(ifft2c_mri(cart[:, :, :, c]@Phi[t]))), isign=-1, eps=1e-6) for c in range(C)])
    # S2: + |k|>1 zero
    s2 = np.stack([finufft.nufft2d2(fx, fy, np.ascontiguousarray(rot(ifft2c_mri(cart_z[:, :, :, c]@Phi[t]))), isign=-1, eps=1e-6) for c in range(C)])
    # S3: full render (SENSE) -> re-expand per coil
    dyn_t = thC@Phi[t]                                                            # coil-combined image (native)
    s3 = np.stack([finufft.nufft2d2(fx, fy, np.ascontiguousarray(rot(b1[:, :, c]*dyn_t)), isign=-1, eps=1e-6) for c in range(C)])
    for nm, S in [("S1", s1), ("S2", s2), ("S3", s3)]:
        sc = np.sum(np.conj(S)*ref)/(np.sum(np.abs(S)**2)+1e-30); Ss = sc*S       # one complex scale / frame
        for b, mk in {**bb, "all": np.ones_like(rr, bool)}.items():
            acc[nm][b][0] += (np.abs(Ss[:, mk]-ref[:, mk])**2).sum(); acc[nm][b][1] += (np.abs(ref[:, mk])**2).sum()
    # SENSE isolation (image domain): combine per-coil zeroed images, re-expand, rel error
    imgs = np.stack([ifft2c_mri(cart_z[:, :, :, c]@Phi[t]) for c in range(C)], -1)  # [RO,RO,C] per-coil images
    rho = np.sum(np.conj(b1)*imgs, -1)/den; reexp = b1*rho[:, :, None]
    sense_num += np.sum(np.abs(reexp-imgs)**2); sense_den += np.sum(np.abs(imgs)**2)

db = lambda n, dd: 10*np.log10((dd+1e-30)/(n+1e-30))
# ---- de-ramp control: does a per-frame sub-pixel shift (linear k-phase) + scale collapse the loss? ----
accd = {s: {b: [0.0, 0.0] for b in ["low(<0.10)", "mid(0.10-0.5)", "high(>0.5)", "all"]} for s in ["S1", "S3"]}
for t in FR:
    fx = np.ascontiguousarray((2*np.pi*kx[t, tr]).reshape(-1).astype(np.float64)); fy = np.ascontiguousarray((2*np.pi*ky[t, tr]).reshape(-1).astype(np.float64))
    rr = np.abs(kx[t, tr]+1j*ky[t, tr]).reshape(-1)/0.5; bb = bands(rr)
    cx = torch.tensor(np.stack([2*kx[t, tr].reshape(-1), 2*ky[t, tr].reshape(-1)], 1), dtype=torch.float32, device=dev)
    tt = torch.full((cx.shape[0],), 2*tq[t]/Ttime-1, dtype=torch.float32, device=dev)
    ref = np.empty((C, cx.shape[0]), np.complex128)
    with torch.no_grad():
        for c in range(C):
            prr = nz.denormalize(cx, m(cx, tt, torch.full((cx.shape[0],), c, dtype=torch.long, device=dev))); ref[c] = (prr[:, 0]+1j*prr[:, 1]).cpu().numpy()
    s1 = np.stack([finufft.nufft2d2(fx, fy, np.ascontiguousarray(rot(ifft2c_mri(cart[:, :, :, c]@Phi[t]))), isign=-1, eps=1e-6) for c in range(C)])
    dyn_t = thC@Phi[t]; s3 = np.stack([finufft.nufft2d2(fx, fy, np.ascontiguousarray(rot(b1[:, :, c]*dyn_t)), isign=-1, eps=1e-6) for c in range(C)])
    for nm, S in [("S1", s1), ("S3", s3)]:
        z = (ref*np.conj(S)).reshape(-1); ph = np.angle(z); w = np.abs(ref).reshape(-1)**2   # weighted linear phase fit
        FX = np.tile(fx, C); FY = np.tile(fy, C); Aw = np.stack([np.ones_like(FX), FX, FY], 1)*np.sqrt(w)[:, None]
        c012, *_ = np.linalg.lstsq(Aw, ph*np.sqrt(w), rcond=None)                 # phi ~ c0 + c1 fx + c2 fy (sub-pixel shift)
        Scorr = S*np.exp(1j*(c012[0]+c012[1]*fx[None]+c012[2]*fy[None]))
        sc = np.sum(np.conj(Scorr)*ref)/(np.sum(np.abs(Scorr)**2)+1e-30); Sc = sc*Scorr
        for b, mk in {**bb, "all": np.ones_like(rr, bool)}.items():
            accd[nm][b][0] += (np.abs(Sc[:, mk]-ref[:, mk])**2).sum(); accd[nm][b][1] += (np.abs(ref[:, mk])**2).sum()
print(f"{'stage':34s} {'low(<0.10)':>12s} {'mid(0.10-0.5)':>14s} {'high(>0.5)':>12s} {'all':>10s}")
print(f"{'ref  amp(spoke)@basis vs measured':34s}  (39.5 dB, per L1)")
labels = {"S1": "S1 grid-query round-trip [c+e+g]", "S2": "S2 + |k|>1 zeroing [d]", "S3": "S3 + SENSE combine/re-expand [f]"}
for s in ["S1", "S2", "S3"]:
    r = acc[s]; print(f"{labels[s]:34s} {db(*r['low(<0.10)']):12.2f} {db(*r['mid(0.10-0.5)']):14.2f} {db(*r['high(>0.5)']):12.2f} {db(*r['all']):10.2f}")
print(f"\n-- after per-frame sub-pixel de-ramp (removes a residual linear k-phase = frame shift) --")
for s in ["S1", "S3"]:
    r = accd[s]; print(f"{('  '+s+' de-ramped'):34s} {db(*r['low(<0.10)']):12.2f} {db(*r['mid(0.10-0.5)']):14.2f} {db(*r['high(>0.5)']):12.2f} {db(*r['all']):10.2f}")
print(f"\nSENSE round-trip in isolation (image domain): {db(sense_num, sense_den):.2f} dB  (~inf => coils in b1 range, not the loss)")
print("-> if de-ramped S1 jumps to >35 dB, the grid-query 'loss' was a sub-pixel FRAME artifact, not real")
print("DONE_L2")
