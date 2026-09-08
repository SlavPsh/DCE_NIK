"""TASK 4: NIK-F0 on the matched-model XCAT sim. Builds the NIK dataset from the synthetic
kept-spoke k-space, trains wire_ff_patlak F0 (aif_xcat = same F0 basis), extracts coefficient
maps via canonical Path C (amplitudes->denorm->IFFT->SENSE) + Path B cross-check. No in-vivo
checkpoint touched. usage: python task4_nik.py --frac f25 --seed 0 --steps 40000"""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, sys, time, numpy as np, torch
from types import SimpleNamespace
sys.path.insert(0, "."); sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import nik_adapter as A
from train_grasp_nik import build_model
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
from nik_focal_loss import composable_kspace_loss
from fftc import ifft2c_mri, crop_img
D = "/scratch/rnga/vvpshenov/DCE_NIK"; OUT = f"{D}/results/task4_xcat_nomotion_pilot"
FIX = dict(hidden=512, depth=12, w0=62.0, s0=15.0, k_freq=256, k_sigma=2.5, t_freq=32, t_sigma=1.5, coil_embed_dim=8, env=0.75)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frac", required=True, choices=["f100", "f25"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--steps", type=int, default=40000); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = np.load(f"{OUT}/arrays/sim.npz"); Phi = S["Phi"]; b1 = S["b1"]; times = S["times"]; kx = S["kx"]; ky = S["ky"]; keep = S["keep_f25"]
    F, NA, RO = kx.shape; C = b1.shape[-1]; Tt = float(times.max()); N = RO
    y = np.load(f"{OUT}/arrays/y{'100' if a.frac=='f100' else '25_in'}.npy", allow_pickle=True)
    kmask = np.ones((F, NA), bool) if a.frac == "f100" else keep
    # build dataset from kept spokes (make_xcat_dataset convention: model coord = 2*traj)
    xs, ys, ts, cs, sp = [], [], [], [], []
    tnorm = (2.0 * times / Tt - 1.0).astype(np.float32)
    for t in range(F):
        kxa = kx[t][kmask[t]].reshape(-1); kya = ky[t][kmask[t]].reshape(-1); m = kxa.size
        yt = np.asarray(y[t]).astype(np.complex64).reshape(C, m)
        for c in range(C):
            xs.append(np.stack([2 * kxa, 2 * kya], 1)); ys.append(np.stack([yt[c].real, yt[c].imag], 1))
            ts.append(np.full(m, tnorm[t], np.float32)); cs.append(np.full(m, c, np.int64)); sp.append(np.full(m, t, np.int64))
    X = torch.tensor(np.concatenate(xs), dtype=torch.float32, device=dev); Y = torch.tensor(np.concatenate(ys), dtype=torch.float32, device=dev)
    T = torch.tensor(np.concatenate(ts), dtype=torch.float32, device=dev); Ct = torch.tensor(np.concatenate(cs), dtype=torch.long, device=dev)
    print(f"frac={a.frac} seed={a.seed}: {X.shape[0]} samples, {F} frames, {int(kmask[0].sum())} angles/frame", flush=True)
    dcf = compute_dcf_radial(X, method="simple_ramp")
    nz = KSpaceNormalizer(); nz.fit(X, Y, dcf=dcf, envelope_exponent=FIX["env"]); Yn = nz.normalize(X, Y)
    torch.manual_seed(a.seed)
    args = SimpleNamespace(model="wire_ff_patlak", patlak_free=0, aif_file=f"{D}/aif_xcat.npz",
        coil_embed_dim=FIX["coil_embed_dim"], hidden=FIX["hidden"], depth=FIX["depth"], w0=FIX["w0"], s0=FIX["s0"],
        k_freq=FIX["k_freq"], k_sigma=FIX["k_sigma"], t_freq=FIX["t_freq"], t_sigma=FIX["t_sigma"], ff_seed=a.seed,
        phi_hidden=64, phi_depth=3, phi_w0=30.0)
    model = build_model(args, C).to(dev); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=3e-3)
    BS = 16384                                                                # fits the ~9.5GB MIG slice
    Ntr = X.shape[0]; t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, Ntr, (BS,), device=dev); opt.zero_grad(set_to_none=True)
        loss = composable_kspace_loss(model(X[idx], T[idx], Ct[idx]), Yn[idx], dcf=torch.ones(BS, device=dev),
                                      use_dcf=False, dcf_power=0.0, use_focal=False, focal_warmup_progress=1.0)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 5000 == 0 or step == a.steps: print(f"  step {step} loss {float(loss):.3e} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    ck = f"{OUT}/checkpoints/nik_F0_{a.frac}_seed{a.seed}.pt"
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(dict(state_dict=sd, model="wire_ff_patlak", patlak_free=0, ncc=C, **{k: FIX[k] for k in ["hidden","depth","w0","s0","k_freq","k_sigma","t_freq","t_sigma","coil_embed_dim"]}, ff_seed=a.seed), ck)
    # ---- Path C: amplitudes -> denorm -> IFFT per coil -> SENSE common maps (chunked for MIG) ----
    coords = torch.from_numpy(A.cartesian_grid(N)).to(dev); rr = np.sqrt(A.cartesian_grid(N)[:,0]**2 + A.cartesian_grid(N)[:,1]**2).reshape(N,N)
    P = coords.shape[0]; CH = 8192
    cart = np.zeros((N, N, 3, C), np.complex64)
    with torch.no_grad():
        for c in range(C):
            out = np.zeros((P, 3), np.complex64)
            for i in range(0, P, CH):
                cco = coords[i:i+CH]; cc = torch.full((cco.shape[0],), c, dtype=torch.long, device=dev); Amp = model.amplitudes(cco, cc)
                for r in range(3):
                    pr = nz.denormalize(cco, Amp[:, r, :].contiguous()); out[i:i+CH, r] = (pr[:,0]+1j*pr[:,1]).cpu().numpy()
            cart[:, :, :, c] = out.reshape(N, N, 3)
    cart[rr > 1.0] = 0
    a_rc = crop_img(ifft2c_mri(cart.reshape(N,N,3*C)).reshape(N,N,3,C), N, N)
    thetaC = np.stack([np.sum(np.conj(b1)*a_rc[:,:,r,:],-1)/(np.sum(np.abs(b1)**2,-1)+1e-8) for r in range(3)], -1)
    # ---- Path B: reconstruct dynamic on the cart grid, project onto Phi ----
    frame_t = tnorm; Ic = np.zeros((N, N, F), np.complex64)
    with torch.no_grad():
        for f in range(F):
            ci = np.zeros((N, N, C), np.complex64)
            for c in range(C):
                out = np.zeros(P, np.complex64)
                for i in range(0, P, CH):
                    cco = coords[i:i+CH]; tf = torch.full((cco.shape[0],), float(frame_t[f]), device=dev)
                    cc = torch.full((cco.shape[0],), c, dtype=torch.long, device=dev)
                    pr = nz.denormalize(cco, model(cco, tf, cc)); out[i:i+CH] = (pr[:,0]+1j*pr[:,1]).cpu().numpy()
                ci[:, :, c] = out.reshape(N, N)
            ci_masked = ci.copy(); ci_masked[rr>1.0]=0; im = ifft2c_mri(ci_masked)
            Ic[:,:,f] = np.sum(np.conj(b1)*im,-1)/(np.sum(np.abs(b1)**2,-1)+1e-8)
    thetaB = np.einsum("rt,xyt->xyr", np.linalg.pinv(Phi), Ic)
    np.savez(f"{OUT}/arrays/nik_F0_{a.frac}_seed{a.seed}.npz", thetaC=thetaC.astype(np.complex64), thetaB=thetaB.astype(np.complex64), Ic=Ic.astype(np.complex64))
    body = S["labels"] > 0
    bc = float(np.linalg.norm(thetaC[body][:,1]-thetaB[body][:,1])/(np.linalg.norm(thetaB[body][:,1])+1e-12))
    print(f"DONE {a.frac} seed{a.seed}: PathB-vs-C intAIF NRMSE {bc:.3e} -> {OUT}/arrays/nik_F0_{a.frac}_seed{a.seed}.npz", flush=True)

if __name__ == "__main__": main()
