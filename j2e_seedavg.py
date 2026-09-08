"""J2e: complex seed-averaging of existing NIK checkpoints (NO training; inference only).
For each config, reload the best checkpoint of each seed, reconstruct the COMPLEX dynamic, apply the
1px-fixed rot180, then compare three combinations vs XCAT truth:
  (a) single-seed  : mean over seeds of per-seed NRMSE
  (b) complex-avg  : NRMSE of |mean_seed(complex recon)|   (cancels incoherent phase/noise)
  (c) magnitude-avg: NRMSE of  mean_seed(|recon|)          (positive-biased, keeps noise floor)
All truth-LS-scaled once (eval convention). check_recon() guards every averaged output."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os
import xph_pipeline as P, xph_common as X, recon_asserts as RA
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1))                     # geometric, commutes w/ averaging
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq)
rv = float(Tr[body].max()-Tr[body].min())                                        # GLOBAL truth range (deliverable convention)
def mnr(a, b, m): return float(np.sqrt(np.mean((a[m]-b[m])**2))/(rv+1e-12))
CK = f"{P.OUT}/checkpoints"

def recon_seed(model_type, rank, width, seed, step):
    tag = ({"wire_ff_subspace": f"sub{rank}", "wire_ff": "free", "patlak": None}[model_type]
           or "") + f"_w{width}_s{seed}" if model_type != "patlak" else f"w{width}_ks2.5_s{seed}"
    p = f"{CK}/{tag}/ck_{step}.pt"
    if not os.path.exists(p):                                                     # fall back to final
        import glob; cs = sorted(glob.glob(f"{CK}/{tag}/ck_*.pt")); p = cs[-1]
    if model_type == "patlak":
        m = P.make_model(width, P.FIX["k_sigma"], seed, C, dev)
    else:
        m = P.make_model_g(model_type, width, P.FIX["k_sigma"], seed, C, dev, rank=rank, warmstart=False)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); m.eval()
    if model_type == "patlak":
        dyn, _ = P.reconstruct_pathC(m, nz, tq, dev)
    else:
        dyn = P.reconstruct_g(m, nz, tq, dev)                                     # complex [RO,RO,F]
    return np.stack([rot(dyn[:, :, t]) for t in range(dyn.shape[2])], -1)         # complex, aligned

def ls_nrmse(recmag, name):
    s = np.sum(recmag[body]*Tr[body])/(np.sum(recmag[body]**2)+1e-12); rec = recmag*s
    RA.check_recon(rec, Tr, mask=body, name=name)
    return np.mean([mnr(rec[:, :, t], Tr[:, :, t], body) for t in range(len(tq))])

CFG = [("sub5", "wire_ff_subspace", 5, [(0, 36000), (1, 40000), (2, 36000)]),
       ("sub12", "wire_ff_subspace", 12, [(0, 40000), (1, 36000), (2, 40000)]),
       ("sub16", "wire_ff_subspace", 16, [(0, 24000), (1, 40000)]),
       ("free", "wire_ff", 0, [(0, 40000), (1, 34000), (2, 36000)]),
       ("F0", "patlak", 0, [(0, 26000), (1, 16000), (2, 24000)])]
print(f"{'config':7s} {'nseed':>5s} {'single-seed':>12s} {'complex-avg':>12s} {'mag-avg':>10s}  {'cplx delta':>10s}")
rows = []
for name, mt, rk, seeds in CFG:
    recs = [recon_seed(mt, rk, 768, s, st) for s, st in seeds]                    # list of complex [RO,RO,F]
    single = np.mean([ls_nrmse(np.abs(r), f"{name}_s{seeds[i][0]}") for i, r in enumerate(recs)])
    cplx = ls_nrmse(np.abs(np.mean(recs, 0)), f"{name}_cplxavg")
    mag = ls_nrmse(np.mean([np.abs(r) for r in recs], 0), f"{name}_magavg")
    dl = 100*(cplx-single)/single
    print(f"{name:7s} {len(seeds):5d} {single:12.4f} {cplx:12.4f} {mag:10.4f}  {dl:+9.1f}%", flush=True)
    rows.append((name, len(seeds), single, cplx, mag))
np.savez(f"{P.OUT}/arrays/j2e_seedavg.npz", configs=[r[0] for r in rows],
         nseed=[r[1] for r in rows], single=[r[2] for r in rows], complex_avg=[r[3] for r in rows], mag_avg=[r[4] for r in rows])
print("DONE_J2E")
