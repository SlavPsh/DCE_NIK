"""K3a (proper): low-|k| CG-SENSE DATA-CONSISTENCY projection warm-started at the render. The render
is only ~10 dB consistent with measured low-|k| (raw network is 40 dB) -> the rank-R/grid/SENSE render
loses DC. Fix it the way GRASP's NLCG does: per frame, run a few CG iterations of min||E_low x - y_low||^2
(E_low = low-|k| radial SENSE forward), warm-started at the render frame, then splice the resulting
data-consistent low-|k| disk back into the render's k-space (high-|k| untouched). Score vs XCAT truth.
Radius sweep 0.05/0.10/0.15. Only measured data + b1 used; truth is eval-only."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob, finufft
import xph_pipeline as P, xph_common as X, recon_asserts as RA
from masked_metrics import haarpsi_masked, ssim_masked
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1))
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"])
kx = d["kx"]; ky = d["ky"]; RO = d["b1"].shape[0]; b1 = d["b1"].astype(np.complex128); F = len(tq)
tr = np.array(P.TRAIN_ANG); rv = float(Tr[body].max()-Tr[body].min()); den = np.sum(np.abs(b1)**2, -1)+1e-8
yy, xxg = np.mgrid[0:RO, 0:RO]; r01g = np.sqrt(((xxg-RO/2)/(RO/2))**2+((yy-RO/2)/(RO/2))**2)
b1c = np.ascontiguousarray(np.transpose(b1, (2, 0, 1)))                            # [C,RO,RO]

def build(kind, s, st):
    tag = f"{'sub16' if kind=='sub16' else 'free'}_w768_s{s}"; p = f"{P.OUT}/checkpoints/{tag}/ck_{st}.pt"
    if not os.path.exists(p): p = sorted(glob.glob(f"{P.OUT}/checkpoints/{tag}/ck_*.pt"))[-1]
    m = P.make_model_g("wire_ff_subspace" if kind == "sub16" else "wire_ff", 768, P.FIX["k_sigma"], 0, C, dev, rank=16, warmstart=False)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); m.eval(); return m

@torch.no_grad()
def tf_complex(m):                                                                 # truth-frame complex render
    dyn = P.reconstruct_g(m, nz, tq, dev); return np.stack([rot(dyn[:, :, t]) for t in range(F)], -1)

def dc_project(dyn_tf, radius, mu, niter=12):
    """REGULARIZED soft low-|k| data consistency in the TRUTH frame (nufft-adjoint(measured) aligns
    with truth): min ||E_low x - y_low||^2 + mu||x - x0||^2, warm-started at the render x0, E normalized
    so mu is in prior/data units. mu large -> stay at render; mu small -> pull to (sparse, streaky) data."""
    out = dyn_tf.copy()
    for t in range(F):
        rr = np.abs(kx[t, tr]+1j*ky[t, tr]).reshape(-1)/0.5; low = rr < radius
        if low.sum() < 8: continue
        x0 = dyn_tf[:, :, t].astype(np.complex128)
        fx = np.ascontiguousarray((2*np.pi*kx[t, tr]).reshape(-1)[low].astype(np.float64)); fy = np.ascontiguousarray((2*np.pi*ky[t, tr]).reshape(-1)[low].astype(np.float64))
        def Efwd(x): return finufft.nufft2d2(fx, fy, np.ascontiguousarray(x[None]*b1c), isign=-1, eps=1e-6)
        def Eadj(yc): cimg = finufft.nufft2d1(fx, fy, np.ascontiguousarray(yc), (RO, RO), isign=1, eps=1e-6); return np.sum(np.conj(b1)*np.transpose(cimg, (1, 2, 0)), -1)/den
        v = np.random.randn(RO, RO)+1j*np.random.randn(RO, RO)                      # power iteration -> ||EHE||
        for _ in range(3): v = Eadj(Efwd(v)); v = v/(np.linalg.norm(v)+1e-30)
        L = float(np.real(np.vdot(v, Eadj(Efwd(v))))+1e-30); s = np.sqrt(L)
        y = np.ascontiguousarray(d["kdata"][:, t, tr, :].reshape(C, -1)[:, low].astype(np.complex128))/s
        En = lambda x: Efwd(x)/s; EnH = lambda yc: Eadj(yc)/s                       # normalized so ||EnH En||~1
        A = lambda x: EnH(En(x))+mu*x; b = EnH(y)+mu*x0                             # (EHE+muI)x = EHy+mu x0
        x = x0.copy(); r = b-A(x); p = r.copy(); rs = np.vdot(r, r).real
        for _ in range(niter):
            Ap = A(p); a = rs/(np.vdot(p, Ap).real+1e-30); x = x+a*p; r = r-a*Ap
            rs2 = np.vdot(r, r).real; p = r+(rs2/(rs+1e-30))*p; rs = rs2
        out[:, :, t] = x
    return out

def score_vol(recmag, name, check=False):
    s = np.sum(recmag[body]*Tr[body])/(np.sum(recmag[body]**2)+1e-12); rec = recmag*s
    if check: RA.check_recon(rec, Tr, mask=body, name=name)
    nrmse = float(np.mean([np.sqrt(np.mean((rec[:, :, t][body]-Tr[:, :, t][body])**2))/rv for t in range(F)]))
    fs = np.arange(0, F, 8); pk = float(Tr[body].max()); mse = float(np.mean([((rec[:, :, t][body]-Tr[:, :, t][body])**2).mean() for t in fs]))
    psnr = 10*np.log10(pk**2/(mse+1e-20)); mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev); hs, ss = [], []
    for t in fs:
        vmax = float(np.percentile(Tr[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        rt = torch.from_numpy(np.clip(Tr[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu())); ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))
    cur = {nm: float(np.linalg.norm(rec[Rz[nm]].mean(0)-Tr[Rz[nm]].mean(0))/(np.linalg.norm(Tr[Rz[nm]].mean(0))+1e-12)) for nm in ("aorta", "cortex", "medulla")}
    bg = float(np.sqrt((rec[~body & (r01g < 0.6)]**2).mean())/(np.sqrt((rec[body]**2).mean())+1e-12))
    return dict(name=name, nrmse=nrmse, psnr=psnr, haarpsi=float(np.mean(hs)), ssim=float(np.mean(ss)), cur=cur, bg=bg)

def fmt(r): return (f"{r['name']:26s} PSNR {r['psnr']:6.2f}  SSIM {r['ssim']:.3f}  Haar {r['haarpsi']:.3f}  NRMSE {r['nrmse']:.4f} | "
                    f"aorta {r['cur']['aorta']:.3f} cortex {r['cur']['cortex']:.3f} medulla {r['cur']['medulla']:.3f} | bg {r['bg']:.4f}")

gk = np.load(f"{P.OUT}/arrays/grasp_ksweep_K12.npz")["rec"].astype(np.float32)
print("REFERENCE:", fmt(score_vol(gk, "GRASP-K12", check=True)), flush=True)
CFG = {"sub16": [(0, 24000), (1, 40000)]}
if not os.environ.get("K3A_QUICK"): CFG["free"] = [(0, 40000), (1, 34000), (2, 36000)]
rows = []
for kind, seeds in CFG.items():
    tfs = [tf_complex(build(kind, s, st)) for s, st in seeds]; print(f"  built {kind} ({len(tfs)} seeds)", flush=True)
    GRID = [(0.10, 3.0), (0.10, 1.0), (0.10, 0.3), (0.05, 1.0), (0.15, 1.0)]        # (radius, mu)
    for i, (s, st) in enumerate(seeds):
        b = score_vol(np.abs(tfs[i]), f"{kind}_s{s} base", check=(i == 0)); print(fmt(b), flush=True); rows.append(b)
        for R, mu in GRID:
            r = score_vol(np.abs(dc_project(tfs[i], R, mu)), f"{kind}_s{s} DC r{R} mu{mu}"); print(fmt(r), f"| dPSNR {r['psnr']-b['psnr']:+.2f}", flush=True); rows.append(r)
    navg = np.mean(tfs, 0); ba = score_vol(np.abs(navg), f"{kind}_cplxavg base", check=True); print(fmt(ba), flush=True); rows.append(ba)
    for R, mu in GRID:
        r = score_vol(np.abs(dc_project(navg, R, mu)), f"{kind}_cplxavg DC r{R} mu{mu}"); print(fmt(r), f"| dPSNR {r['psnr']-ba['psnr']:+.2f}", flush=True); rows.append(r)
import csv
with open(f"{P.OUT}/k3a_dc.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["name", "psnr", "ssim", "haarpsi", "nrmse", "aorta", "cortex", "medulla", "bg"])
    for r in rows: w.writerow([r["name"], f"{r['psnr']:.3f}", f"{r['ssim']:.4f}", f"{r['haarpsi']:.4f}", f"{r['nrmse']:.5f}", f"{r['cur']['aorta']:.4f}", f"{r['cur']['cortex']:.4f}", f"{r['cur']['medulla']:.4f}", f"{r['bg']:.5f}"])
print("DONE_K3A_DC")
