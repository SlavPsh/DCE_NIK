"""Step 2b: regenerate all 5 NIK variants (sub5, sub12, sub16, free, F0) on XCAT z15 with the CORRECTED
centered rot180, calling check_recon() on every output (RAISES on any geometry/orientation/finite
failure -> job halts, never worked around). Updates rec_best (+ derived fields) in each npz; the
aggregate recomputes all metrics from rec_best. Prints old vs new SSIM and the A1 shift per variant
(2a verification: F0 shift must be ~0)."""
import warnings; warnings.filterwarnings("ignore")
import glob, numpy as np, torch
from scipy.ndimage import uniform_filter
import xph_pipeline as P, xph_common as X, recon_asserts as RA
OUT = X.OUT; A = f"{OUT}/arrays"; dev = torch.device("cuda")
d = P.data(); tq = d["times"]; body = d["labels"] > 0; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq); rv = float(Tr[body].max()-Tr[body].min())
_, _, _, _, nz, dims = P.build_train(dev); C = dims[3]
FIXROT = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1))
def lss(rec): return rec * float((rec[body]*Tr[body]).sum()/((rec[body]**2).sum()+1e-12))
def mnr(a, b): return float(np.sqrt(np.mean((a[body]-b[body])**2))/(rv+1e-12))
def ssim(a, b, win=7):
    C1 = (0.01*rv)**2; C2 = (0.03*rv)**2; ma = uniform_filter(a, win); mb = uniform_filter(b, win)
    va = uniform_filter(a*a, win)-ma**2; vb = uniform_filter(b*b, win)-mb**2; vab = uniform_filter(a*b, win)-ma*mb
    return float((((2*ma*mb+C1)*(2*vab+C2))/((ma**2+mb**2+C1)*(va+vb+C2)))[body].mean())
def ssim_vol(v): return float(np.mean([ssim(v[:, :, t], Tr[:, :, t]) for t in range(v.shape[2])]))

jobs = []
for f in sorted(glob.glob(f"{A}/img_eval_sub*_w768_s*.npz")) + sorted(glob.glob(f"{A}/img_eval_free_w768_s*.npz")):
    e = np.load(f, allow_pickle=True); jobs.append(dict(f=f, kind="img", model=str(e["model"]), rank=int(e["rank"]), width=int(e["width"]), tag=str(e["tag"]), best_step=int(e["best_step"])))
for f in sorted(glob.glob(f"{A}/nik_eval_w768_ks*_s*.npz")):
    e = np.load(f, allow_pickle=True); jobs.append(dict(f=f, kind="f0", width=int(e["width"]), ks=float(e["k_sigma"]), seed=int(e["seed"]), tag=str(e["tag"]), best_step=int(e["best_step"])))

print(f"regenerating {len(jobs)} variant/seed recons with fixed rot + check_recon\n")
for j in jobs:
    model = (P.make_model_g(j["model"], j["width"], P.FIX["k_sigma"], 0, C, dev, rank=j["rank"], warmstart=False)
             if j["kind"] == "img" else P.make_model(j["width"], j["ks"], j["seed"], C, dev)); model.eval()
    ck = [p for p in glob.glob(f"{OUT}/checkpoints/{j['tag']}/ck_*.pt") if int(p.split('ck_')[-1].split('.')[0]) == j["best_step"]][0]
    model.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state_dict"])
    dyn = P.reconstruct_g(model, nz, tq, dev) if j["kind"] == "img" else P.reconstruct_pathC(model, nz, tq, dev)[0]
    rec = lss(np.abs(np.stack([FIXROT(dyn[:, :, t]) for t in range(dyn.shape[2])], -1)))
    res = RA.check_recon(rec, Tr, mask=body, name=j["tag"])          # RAISES -> halt on any failure
    sh = res["A1"]["shift"]
    old = dict(np.load(j["f"], allow_pickle=True)); ss_old = ssim_vol(np.abs(old["rec_best"]).astype(np.float64))
    per = np.array([mnr(rec[:, :, t], Tr[:, :, t]) for t in range(len(tq))])
    cur = {nm: float(np.linalg.norm(rec[R[nm]].mean(0)-Tr[R[nm]].mean(0))/(np.linalg.norm(Tr[R[nm]].mean(0))+1e-12)) for nm in ["aorta", "cortex", "medulla"]}
    old["rec_best"] = rec.astype(np.float32); old["per_frame_best"] = per; old["img_nrmse_mean_best"] = float(per.mean())
    old["cur_nrmse_best"] = np.array([cur[k] for k in ("aorta", "cortex", "medulla")])
    np.savez(j["f"], **old)
    print(f"{j['tag']:24s} A1 shift ({sh[0]:+.3f},{sh[1]:+.3f}) A4 id A5 ok | SSIM {ss_old:.4f} -> {ssim_vol(rec):.4f}", flush=True)
print("\nSTEP2_REGEN_DONE")
