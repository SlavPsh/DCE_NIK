"""L3 re-baseline: re-render every NIK config/seed from existing checkpoints through the FIXED
(oversampled) render, check_recon on every output (halts on geometry failure = the verify step),
score spatial (PSNR/SSIM/HaarPSI/NRMSE) AND temporal (aorta/cortex/medulla curve NRMSE + aorta
peak/TTP/FWHM), report per-config mean +/- half-spread over seeds, plus complex seed-average (J2e),
all vs GRASP-K12. Deliverable numbers use the global-range NRMSE (rv). No training; truth eval-only."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob, csv
import xph_pipeline as P, xph_common as X, recon_asserts as RA
import os as _os
# phantom reference-method plumbing. defaults = grasp-pro (unchanged). grasp v2:
#   GRASP_NPZ=grasp_v2_recon.npz GRASP_LABEL=GRASP-v2 TAG=_gv2
_GNPZ = _os.environ.get("GRASP_NPZ", "grasp_ksweep_K12.npz")
_GLAB = _os.environ.get("GRASP_LABEL", "GRASP-K12 (ref)")
_TAG = _os.environ.get("TAG", "")

from masked_metrics import haarpsi_masked, ssim_masked
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"])
RO = d["b1"].shape[0]; F = len(tq); rv = float(Tr[body].max()-Tr[body].min())
rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1)); HF = np.arange(0, F, 2)   # HaarPSI/SSIM frame subset
print(f"OVERSAMPLE = {P.OVERSAMPLE}", flush=True)

def load(kind, s, st, rank):
    tag = ({"sub": f"sub{rank}", "free": "free", "F0": None}[kind]) or ""
    tag = (f"sub{rank}_w768_s{s}" if kind == "sub" else (f"free_w768_s{s}" if kind == "free" else f"w768_ks2.5_s{s}"))
    p = f"{P.OUT}/checkpoints/{tag}/ck_{st}.pt"
    if not os.path.exists(p): p = sorted(glob.glob(f"{P.OUT}/checkpoints/{tag}/ck_*.pt"))[-1]
    if kind == "F0":
        m = P.make_model(768, P.FIX["k_sigma"], s, C, dev)
    else:
        m = P.make_model_g("wire_ff_subspace" if kind == "sub" else "wire_ff", 768, P.FIX["k_sigma"], 0, C, dev, rank=rank, warmstart=False)
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=False)["state_dict"]); m.eval(); return m, kind

@torch.no_grad()
def render_tf(mk):
    m, kind = mk
    dyn = (P.reconstruct_pathC(m, nz, tq, dev)[0] if kind == "F0" else P.reconstruct_g(m, nz, tq, dev))
    return np.stack([rot(dyn[:, :, t]) for t in range(F)], -1)                     # truth-frame complex

def score(dyn_tf, name):
    rec = np.abs(dyn_tf); s = np.sum(rec[body]*Tr[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s
    RA.check_recon(rec, Tr, mask=body, name=name)                                  # VERIFY (halts on geometry failure)
    nrmse = float(np.mean([np.sqrt(np.mean((rec[:, :, t][body]-Tr[:, :, t][body])**2))/rv for t in range(F)]))
    pk = float(Tr[body].max()); mse = float(np.mean([((rec[:, :, t][body]-Tr[:, :, t][body])**2).mean() for t in range(F)]))
    psnr = 10*np.log10(pk**2/(mse+1e-20)); mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev); hs, ss = [], []
    for t in HF:
        vmax = float(np.percentile(Tr[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        rt = torch.from_numpy(np.clip(Tr[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu())); ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))
    cur = {nm: float(np.linalg.norm(rec[Rz[nm]].mean(0)-Tr[Rz[nm]].mean(0))/(np.linalg.norm(Tr[Rz[nm]].mean(0))+1e-12)) for nm in ("aorta", "cortex", "medulla")}
    ac = rec[Rz["aorta"]].mean(0); at = Tr[Rz["aorta"]].mean(0)
    return dict(psnr=psnr, ssim=float(np.mean(ss)), haarpsi=float(np.mean(hs)), nrmse=nrmse, **{f"c_{k}": v for k, v in cur.items()},
                aorta_ttp=float(tq[np.argmax(ac)]), aorta_ttp_t=float(tq[np.argmax(at)]), aorta_pk=float(ac.max()), aorta_pk_t=float(at.max()))

KEYS = ["psnr", "ssim", "haarpsi", "nrmse", "c_aorta", "c_cortex", "c_medulla", "aorta_ttp", "aorta_pk"]
CFG = [("sub", 16, [(0, 24000), (1, 40000)]), ("sub", 5, [(0, 36000), (1, 40000), (2, 36000)]),
       ("sub", 12, [(0, 40000), (1, 36000), (2, 40000)]), ("free", 0, [(0, 40000), (1, 34000), (2, 36000)]),
       ("F0", 0, [(0, 26000), (1, 16000), (2, 24000)])]
gk = np.abs(np.load(f"{P.OUT}/arrays/{_GNPZ}")["rec"]).astype(np.float32)
sgk = np.sum(gk[body]*Tr[body])/(np.sum(gk[body]**2)+1e-12)
rows = []
def emit(name, r): rows.append((name, r)); print(f"{name:20s} PSNR {r['psnr']:6.2f} SSIM {r['ssim']:.4f} Haar {r['haarpsi']:.4f} NRMSE {r['nrmse']:.4f} | aortaC {r['c_aorta']:.3f} cortexC {r['c_cortex']:.3f} medullaC {r['c_medulla']:.3f}", flush=True)
# GRASP reference (score gk directly)
gkr = np.stack([gk[:, :, t] for t in range(F)], -1)*sgk
class _o:
    pass
def score_mag(rec, name):
    RA.check_recon(rec, Tr, mask=body, name=name)
    nrmse = float(np.mean([np.sqrt(np.mean((rec[:, :, t][body]-Tr[:, :, t][body])**2))/rv for t in range(F)]))
    pk = float(Tr[body].max()); mse = float(np.mean([((rec[:, :, t][body]-Tr[:, :, t][body])**2).mean() for t in range(F)]))
    psnr = 10*np.log10(pk**2/(mse+1e-20)); mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev); hs, ss = [], []
    for t in HF:
        vmax = float(np.percentile(Tr[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev); rt = torch.from_numpy(np.clip(Tr[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu())); ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))
    cur = {nm: float(np.linalg.norm(rec[Rz[nm]].mean(0)-Tr[Rz[nm]].mean(0))/(np.linalg.norm(Tr[Rz[nm]].mean(0))+1e-12)) for nm in ("aorta", "cortex", "medulla")}
    # same aorta curve definition as score(); was hardcoded 0 which read as "no bolus" in the table
    ac = rec[Rz["aorta"]].mean(0)
    return dict(psnr=psnr, ssim=float(np.mean(ss)), haarpsi=float(np.mean(hs)), nrmse=nrmse, c_aorta=cur["aorta"], c_cortex=cur["cortex"], c_medulla=cur["medulla"], aorta_ttp=float(tq[np.argmax(ac)]), aorta_pk=float(ac.max()))
emit(_GLAB, score_mag(gkr, "GRASP"))
for kind, rank, seeds in CFG:
    name = f"{kind}{rank}" if kind == "sub" else kind
    tfs = [render_tf(load(kind, s, st, rank)) for s, st in seeds]
    per = [score(t, f"{name}_s{seeds[i][0]}") for i, t in enumerate(tfs)]
    mu = {k: np.mean([p[k] for p in per]) for k in KEYS}; sp = {k: (np.max([p[k] for p in per])-np.min([p[k] for p in per]))/2 for k in KEYS}
    print(f"\n-- {name} (n={len(seeds)} seeds) mean +/- half-spread --", flush=True)
    emit(f"{name} mean", mu); print(f"  spread: PSNR +/-{sp['psnr']:.2f} SSIM +/-{sp['ssim']:.4f} Haar +/-{sp['haarpsi']:.4f} NRMSE +/-{sp['nrmse']:.4f}", flush=True)
    cavg = score(np.mean(tfs, 0), f"{name}_cplxavg"); emit(f"{name} cplx-seedavg", cavg)
with open(f"{P.OUT}/l3_rebaseline{_TAG}.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["name"]+KEYS)
    for name, r in rows: w.writerow([name]+[f"{r.get(k, 0):.4f}" for k in KEYS])
print("\nDONE_L3REBASE")
