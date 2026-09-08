"""SENSE-forward NIK (approach A), phantom. The network is an IMAGE INR of the combined object:
amplitudes(x) -> rank-R complex image coefficients c_r(x); rho(x,t)=sum_r c_r(x) phi_r(t). The KNOWN
coil sensitivities are applied in image domain (exact, pointwise): coil image = b1_c . rho. Data loss
= || DFT(b1_c . rho) at measured spokes - measured ||, so coils are b1-consistent by construction
(no coil embedding). Forward is an exact direct DFT via torch matmul (GPU, autograd-native): no NUFFT,
no gridding, no frame-convention pitfall. Compare vs sub16 (2x render) and truth. check_recon gates.
usage: python sense_forward_train.py --steps 12000 --rank 16 --seed 0 --smoke 0"""
import warnings; warnings.filterwarnings("ignore")
import argparse, time, os, numpy as np, torch
from types import SimpleNamespace
import xph_pipeline as P, xph_common as X, recon_asserts as RA
from train_grasp_nik import build_model
from masked_metrics import haarpsi_masked, ssim_masked

def build_image_inr(width, rank, seed, w0, s0, imsig, kfreq, dev):
    """subspace model retuned as an IMAGE INR: low WIRE w0 + low Fourier-feature sigma so the
    coordinate net is SMOOTH (spectral bias toward low spatial freq), unlike the k-space-tuned net."""
    torch.manual_seed(seed)
    args = SimpleNamespace(model="wire_ff_subspace", patlak_free=0, aif_file=P.AIF, hidden=int(width), ff_seed=seed,
                           rank=int(rank), phi_hidden=64, phi_depth=3, phi_w0=30.0, phi_ortho=False, n_pk=-1, radial_alpha=1.0,
                           k_sigma=float(imsig), depth=P.FIX["depth"], w0=float(w0), s0=float(s0), k_freq=int(kfreq),
                           t_freq=P.FIX["t_freq"], t_sigma=P.FIX["t_sigma"], coil_embed_dim=P.FIX["coil_embed_dim"])
    return build_model(args, 1).to(dev)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--fbatch", type=int, default=6); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w0", type=float, default=20.0); ap.add_argument("--s0", type=float, default=10.0)
    ap.add_argument("--imsig", type=float, default=3.0); ap.add_argument("--kfreq", type=int, default=128)
    ap.add_argument("--tag", default=""); ap.add_argument("--smoke", type=int, default=0); a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = P.data(); tq = d["times"]; F = len(tq); RO = d["b1"].shape[0]; C = d["b1"].shape[-1]
    tr = np.array(P.TRAIN_ANG); body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"])
    rv = float(Tr[body].max()-Tr[body].min()); Ttime = float(tq.max())
    b1 = torch.tensor(d["b1"], dtype=torch.complex64, device=dev).reshape(-1, C)              # [RO^2, C]
    # image pixel coords (centered) for the DFT, and INR input coords in [-1,1]
    ax = (np.arange(RO) - RO/2.0)
    xn, yn = np.meshgrid(ax, ax, indexing="ij")
    xn = torch.tensor(xn.reshape(-1), dtype=torch.float32, device=dev); yn = torch.tensor(yn.reshape(-1), dtype=torch.float32, device=dev)
    ic = (np.stack(np.meshgrid((np.arange(RO)-RO//2)/(RO//2), (np.arange(RO)-RO//2)/(RO//2), indexing="ij"), -1)).reshape(-1, 2)
    icoords = torch.tensor(ic, dtype=torch.float32, device=dev)                                # [RO^2,2] INR input
    # measured per-frame spokes + global normalization
    g = 1.0 / (np.abs(d["kdata"][:, :, tr, :]).max() + 1e-12)
    Ymeas = [torch.tensor((d["kdata"][:, t, tr, :].reshape(C, -1).T * g).astype(np.complex64), device=dev) for t in range(F)]  # [M,C]
    FX = [torch.tensor(d["kx"][t, tr].reshape(-1), dtype=torch.float32, device=dev) for t in range(F)]  # cycles/pixel
    FY = [torch.tensor(d["ky"][t, tr].reshape(-1), dtype=torch.float32, device=dev) for t in range(F)]
    TWO_PI = 2*np.pi; DFT_SCALE = [1.0]                                                        # calibrated at init
    def dft_raw(imgC, t):
        ph = torch.exp(-1j*TWO_PI*(FX[t][:, None]*xn[None, :] + FY[t][:, None]*yn[None, :]))    # [M,RO^2] complex
        return ph @ imgC
    def dft_at(imgC, t): return dft_raw(imgC, t)*DFT_SCALE[0]
    # image INR retuned for SMOOTHNESS (low w0/sigma), not the k-space-tuned net (which -> noise)
    model = build_image_inr(a.width, a.rank, a.seed, a.w0, a.s0, a.imsig, a.kfreq, dev); model.train()
    print(f"  image-INR freq: w0={a.w0} s0={a.s0} imsig={a.imsig} kfreq={a.kfreq}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    def coeffs():                                                                              # [RO^2,R] complex, differentiable
        amp = model.amplitudes(icoords, torch.zeros(icoords.shape[0], dtype=torch.long, device=dev))  # [P,R,2]
        return amp[:, :, 0] + 1j*amp[:, :, 1]
    def basis():                                                                               # [F,R] real
        tn = torch.tensor([2*tq[t]/Ttime-1 for t in range(F)], dtype=torch.float32, device=dev)
        return model.basis(tn)[:, :, 0].real.float()
    with torch.no_grad():                                                                      # calibrate DFT scale to init so A(img)~Ymeas (fix conditioning)
        cf0 = coeffs(); Phi0 = basis(); num = 0.0; dn = 0.0
        for t in [F//4, F//2, 3*F//4]:
            rho0 = cf0 @ Phi0[t].to(cf0.dtype); yp0 = dft_raw(b1*rho0[:, None], t)
            num += float(torch.sqrt(torch.mean(torch.abs(yp0)**2))); dn += float(torch.sqrt(torch.mean(torch.abs(Ymeas[t])**2)))
        DFT_SCALE[0] = dn/(num+1e-30); print(f"  DFT_SCALE calibrated = {DFT_SCALE[0]:.3e}", flush=True)
    t0 = time.time(); rng = np.random.RandomState(a.seed)
    for step in range(1, a.steps+1):
        opt.zero_grad(); cf = coeffs(); Phi = basis(); loss = 0.0
        for t in rng.choice(F, a.fbatch, replace=False):
            rho = cf @ Phi[t].to(cf.dtype)                                                     # [RO^2] complex combined image
            imgC = b1 * rho[:, None]                                                           # [RO^2,C] coil images
            yp = dft_at(imgC, int(t))
            loss = loss + torch.mean(torch.abs(yp - Ymeas[int(t)])**2)
        loss = loss / a.fbatch; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % max(1, a.steps//20) == 0 or step == 1:
            print(f"  step {step:6d} loss {float(loss):.4e} ({time.time()-t0:.0f}s)", flush=True)
    # ---- eval: render combined image for all frames, compare to truth ----
    model.eval()
    with torch.no_grad():
        cf = coeffs(); Phi = basis()
        dyn = torch.stack([(cf @ Phi[t].to(cf.dtype)).reshape(RO, RO) for t in range(F)], -1).cpu().numpy()  # [RO,RO,F] complex
    ckdir = f"{P.OUT}/checkpoints/senseA{a.tag}_r{a.rank}_s{a.seed}"; os.makedirs(ckdir, exist_ok=True)   # save FIRST (eval-proof)
    torch.save(dict(state_dict=model.state_dict(), rank=a.rank, seed=a.seed, width=a.width), f"{ckdir}/ck_final.pt")
    # robust registration to truth: the radial forward has a rot180/even-grid ambiguity, so search the
    # full D4 group x integer shift and pick the transform that best matches truth (magnitude, mean frame)
    D4 = {"id": lambda x: x, "rot90": lambda x: np.rot90(x, 1), "rot180": lambda x: np.rot90(x, 2), "rot270": lambda x: np.rot90(x, 3),
          "fliplr": lambda x: x[:, ::-1], "flipud": lambda x: x[::-1, :], "T": lambda x: x.T, "antiT": lambda x: x[::-1, ::-1].T}
    mf = np.abs(dyn).mean(-1); trm = Tr.mean(-1); best = (-2.0, "id", (0, 0))
    for nm, fn in D4.items():
        g = np.ascontiguousarray(fn(mf))
        if g.shape != trm.shape: continue
        for dyi in range(-3, 4):
            for dxi in range(-3, 4):
                gg = np.roll(g, (dyi, dxi), (0, 1)); cc = float(np.corrcoef(gg[body], trm[body])[0, 1])
                if cc > best[0]: best = (cc, nm, (dyi, dxi))
    cc, nm, (dyi, dxi) = best; print(f"  registration: D4={nm} shift=({dyi},{dxi}) corr={cc:.3f}", flush=True)
    fn = D4[nm]; rec = np.abs(np.stack([np.roll(fn(dyn[:, :, t]), (dyi, dxi), (0, 1)) for t in range(F)], -1))
    s = np.sum(rec[body]*Tr[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s
    try: RA.check_recon(rec, Tr, mask=body, name=f"senseA_s{a.seed}"); print("  check_recon PASS", flush=True)
    except Exception as e: print(f"  check_recon NOTE (registered, residual sub-pixel): {e}", flush=True)
    def scorevol(rec):
        nrmse = float(np.mean([np.sqrt(np.mean((rec[:, :, t][body]-Tr[:, :, t][body])**2))/rv for t in range(F)]))
        pk = float(Tr[body].max()); mse = float(np.mean([((rec[:, :, t][body]-Tr[:, :, t][body])**2).mean() for t in range(0, F, 2)]))
        psnr = 10*np.log10(pk**2/(mse+1e-20)); mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev); hs, ss = [], []
        for t in range(0, F, 2):
            vmax = float(np.percentile(Tr[:, :, t][body], 99.5))
            pt = torch.from_numpy(np.clip(rec[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev); rt = torch.from_numpy(np.clip(Tr[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
            hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu())); ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))
        cur = {nm: float(np.linalg.norm(rec[Rz[nm]].mean(0)-Tr[Rz[nm]].mean(0))/(np.linalg.norm(Tr[Rz[nm]].mean(0))+1e-12)) for nm in ("aorta", "cortex", "medulla")}
        return psnr, float(np.mean(ss)), float(np.mean(hs)), nrmse, cur
    psnr, ss, hs, nr, cur = scorevol(rec)
    print(f"\nSENSE-A rank{a.rank} seed{a.seed}: PSNR {psnr:.2f} SSIM {ss:.4f} HaarPSI {hs:.4f} NRMSE {nr:.4f} | aortaC {cur['aorta']:.3f} cortexC {cur['cortex']:.3f} medullaC {cur['medulla']:.3f}", flush=True)
    print("baseline sub16(2x): PSNR 36.42 SSIM 0.9205 HaarPSI 0.8871 NRMSE 0.0148 | aortaC 0.156 | GRASP HaarPSI 0.8707", flush=True)
    if not a.smoke:
        np.savez(f"{P.OUT}/arrays/senseA{a.tag}_r{a.rank}_s{a.seed}.npz", rec=rec.astype(np.float32), psnr=psnr, ssim=ss, haarpsi=hs, nrmse=nr,
                 cur=np.array([cur['aorta'], cur['cortex'], cur['medulla']]))
    print("DONE_SENSEA", flush=True)

if __name__ == "__main__": main()
