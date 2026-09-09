"""step 5 phantom eval, identical pipeline for Patlak (control) and Tofts arms. usage: --sim nomotion|motion"""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, sys, json, glob
ap = argparse.ArgumentParser(); ap.add_argument("--sim", default="nomotion"); a = ap.parse_args()
os.environ["XPH_SIM"] = a.sim
import numpy as np, torch
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import xph_pipeline as P, xph_common as X
from masked_metrics import haarpsi_masked, ssim_masked
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = P.OUT; A = f"{OUT}/arrays"; SW = f"{OUT}/v2_sweep"; RES = "/net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak"
d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); F = len(tq); Rz = X.rois(P.ZI, d["labels"])
G = 5                                                        # 25 spf single setting: 344 -> 68 frames (time-matched CS grid)
EDGES = np.linspace(0, 1, 17)
MT = torch.from_numpy(body.astype(np.float32))[None, None].to(dev)

def winavg(v, G): n = v.shape[-1] // G; return np.stack([v[:, :, g*G:(g+1)*G].mean(2) for g in range(n)], -1)
def ls_scale(img, ref): return img * (np.sum(img[body]*ref[body]) / (np.sum(img[body]**2) + 1e-12))
def img_metrics(img, ref):
    nt = img.shape[-1]; acc = dict(haarpsi=[], ssim=[]); rv = float(ref[body].max()-ref[body].min()); pk = float(ref[body].max())
    for t in range(0, nt, max(1, nt // 40)):
        vmax = float(np.percentile(ref[:, :, t][body], 99.5)) + 1e-12
        x = torch.from_numpy(np.clip(img[:, :, t]/vmax, 0, 1)[None, None]).float().to(dev); y = torch.from_numpy(np.clip(ref[:, :, t]/vmax, 0, 1)[None, None]).float().to(dev)
        acc["haarpsi"].append(float(haarpsi_masked(x, y, MT, data_range=1.0).cpu())); acc["ssim"].append(float(ssim_masked(x, y, MT, data_range=1.0).cpu()))
    mse = float(np.mean([((img[:, :, t][body]-ref[:, :, t][body])**2).mean() for t in range(nt)]))
    nrmse = float(np.mean([np.sqrt(np.mean((img[:, :, t][body]-ref[:, :, t][body])**2))/rv for t in range(nt)]))
    g = lambda v: float(np.sqrt(sum(q**2 for q in np.gradient(v.mean(-1)))[body].mean()))
    return dict(haarpsi=float(np.mean(acc["haarpsi"])), ssim=float(np.mean(acc["ssim"])), psnr=float(10*np.log10(pk**2/(mse+1e-20))), nrmse=nrmse,
                sharpness_rel=g(img)/(g(ref)+1e-12), bg_noise=float(img.mean(-1)[~body].std()))
def fwhm(t, c):
    c = c - np.median(c[:8]); i = int(np.argmax(c)); h = c[i]/2; l = i; r = i
    while l > 0 and c[l] > h: l -= 1
    while r < len(c)-1 and c[r] > h: r += 1
    return float(t[r]-t[l]), float(t[i]), float(c[i])
def curve_metrics(v, t):
    r = {}
    for roi in ("aorta", "cortex", "medulla"):
        c = np.interp(tq, t, np.array([v[..., i][Rz[roi]].mean() for i in range(v.shape[-1])])); ct = Tr[Rz[roi]].mean(0)
        r[f"cur_{roi}"] = float(np.linalg.norm(c-ct)/np.linalg.norm(ct))
        if roi == "aorta":
            fw, tp, pk = fwhm(tq, c); fwt, tpt, pkt = fwhm(tq, ct)
            r.update(aorta_peak_err_pct=100*(pk-pkt)/pkt, aorta_ttp_err_s=tp-tpt, aorta_fwhm_s=fw, aorta_fwhm_truth_s=fwt)
    return r
@torch.no_grad()
def annuli(model, nz, mask):
    X_, Y_, T_, C_, R, _ = P._dataset(mask, dev); n = X_.shape[0]; num = np.zeros(16); den = np.zeros(16)
    CH = 40000
    for i in range(0, n, CH):
        pr = nz.denormalize(X_[i:i+CH], model(X_[i:i+CH], T_[i:i+CH], C_[i:i+CH]))
        yh = (pr[:, 0] + 1j*pr[:, 1]).cpu().numpy(); yt = (Y_[i:i+CH, 0] + 1j*Y_[i:i+CH, 1]).cpu().numpy()
        e = np.abs(yh-yt)**2; p = np.abs(yt)**2; b = np.clip(np.digitize(np.asarray(R[i:i+CH]), EDGES)-1, 0, 15)
        for k in range(16): m = b == k; num[k] += e[m].sum(); den[k] += p[m].sum()
    return dict(nmse=float(num.sum()/den.sum()), annuli=(num/(den+1e-30)).tolist())

rows = []; _, _, _, _, nz, dims = P.build_train(dev); Cc = dims[3]; mk = P.masks()
Tw = winavg(Tr, G)
for arm in ("patlak", "tofts"):
    for s in (0, 1, 2):
        tag = f"w768_ks2.5_s{s}" + ("_tofts16" if arm == "tofts" else ""); f = f"{A}/nik_eval_{tag}.npz"
        if not os.path.exists(f): print(f"  {tag}: MISSING (run incomplete)"); rows.append(dict(arm=arm, seed=s, tag=tag, status="missing")); continue
        ev = np.load(f, allow_pickle=True); rec = ev["rec_best"].astype(np.float32); bstep = int(ev["best_step"])
        ck = torch.load(f"{OUT}/checkpoints/{tag}/ck_{bstep:05d}.pt", map_location=dev, weights_only=False)
        model = (P.make_model_tofts(768, 2.5, s, Cc, dev, ck.get("basis_file") if os.path.exists(str(ck.get("basis_file"))) else None) if arm == "tofts" else P.make_model(768, 2.5, s, Cc, dev))
        model.load_state_dict(ck["state_dict"]); model.eval()
        r = dict(arm=arm, seed=s, tag=tag, status="complete", frames=F, best_step=bstep, rank=int(model.rank), params=int(ck["params"]["total"]),
                 wall_s=float(ck.get("wall_s", np.nan)), peak_gpu_mb=float(ck.get("peak_gpu_mb", np.nan)), val_nmse=float(ev["val_nmse"][int(ev["best_idx"])]))
        r.update({f"fine_{k}": v for k, v in img_metrics(ls_scale(rec, Tr), Tr).items()})
        r.update({f"b68_{k}": v for k, v in img_metrics(ls_scale(winavg(rec, G), Tw), Tw).items()})
        r.update(curve_metrics(rec, tq)); hv = annuli(model, nz, mk["val"]); ht = annuli(model, nz, mk["test"])
        r.update(val_kNMSE=hv["nmse"], test_kNMSE=ht["nmse"], test_annuli=ht["annuli"], val_annuli=hv["annuli"])
        del model; torch.cuda.empty_cache()
        rows.append(r); print(f"  {tag}: ssim68 {r['b68_ssim']:.4f} haar68 {r['b68_haarpsi']:.4f} psnr68 {r['b68_psnr']:.2f} | cortex {r['cur_cortex']:.4f} medulla {r['cur_medulla']:.4f} aorta {r['cur_aorta']:.4f} | test kNMSE {ht['nmse']:.3e}", flush=True)
cs = f"{SW}/v2_G05.npy"
if os.path.exists(cs):
    v = np.abs(np.load(cs)).astype(np.float32); e = np.linspace(0, tq[-1], v.shape[-1]+1); tc = 0.5*(e[:-1]+e[1:])
    r = dict(arm="GRASP-v2 25spf lam0.25 (5 train angles, 68 fr)", seed=-1, tag="v2_G05", status="complete", frames=int(v.shape[-1]))
    r.update({f"b68_{k}": vv for k, vv in img_metrics(ls_scale(v, Tw), Tw).items()}); r.update(curve_metrics(ls_scale(v, Tw), tc)); rows.append(r)
    print(f"  CS ref: ssim68 {r['b68_ssim']:.4f} haar68 {r['b68_haarpsi']:.4f} cortex {r['cur_cortex']:.4f}")
else: rows.append(dict(arm="GRASP-v2 25spf lam0.25", status="blocked: no CS recon for this sim"))
json.dump(rows, open(f"{RES}/phantom_{a.sim}.json", "w"), indent=1)

# table: mean +- sd per arm, delta tofts - patlak
keys = ["b68_ssim", "b68_psnr", "b68_haarpsi", "b68_nrmse", "fine_ssim", "fine_psnr", "fine_haarpsi", "fine_nrmse", "cur_aorta", "cur_cortex", "cur_medulla",
        "aorta_peak_err_pct", "aorta_ttp_err_s", "aorta_fwhm_s", "val_kNMSE", "test_kNMSE", "wall_s", "peak_gpu_mb", "params"]
def ms(arm, k):
    v = np.array([r[k] for r in rows if r.get("arm") == arm and r.get("status") == "complete" and k in r], float); return (np.nanmean(v), np.nanstd(v), v.size) if v.size else (np.nan, np.nan, 0)
lines = [f"# phantom {a.sim} (z15): Patlak (rank 3) vs Tofts (rank 16), seeds 0,1,2, 5 train angles, 40k steps, best-VAL ckpt", "",
         "| metric | Patlak mean±SD (n) | Tofts mean±SD (n) | Δ Tofts−Patlak | CS ref (GRASP-v2 25spf lam0.25, NOT truth) |", "|---|---|---|---|---|"]
for k in keys:
    p, t = ms("patlak", k), ms("tofts", k); c = [r.get(k) for r in rows if str(r.get("arm", "")).startswith("GRASP")]; c = c[0] if c and c[0] is not None else np.nan
    lines.append(f"| {k} | {p[0]:.4g} ± {p[1]:.2g} ({p[2]}) | {t[0]:.4g} ± {t[1]:.2g} ({t[2]}) | {t[0]-p[0]:+.4g} | {c:.4g} |")
lines += ["", "test-angle k-space NMSE per |k| annulus (16 bins, r=|k|/kmax):", "| annulus | Patlak | Tofts |", "|---|---|---|"]
pa = np.array([r["test_annuli"] for r in rows if r.get("arm") == "patlak" and r.get("status") == "complete"]); ta = np.array([r["test_annuli"] for r in rows if r.get("arm") == "tofts" and r.get("status") == "complete"])
if pa.size and ta.size:
    for i in range(16): lines.append(f"| {EDGES[i]:.2f}-{EDGES[i+1]:.2f} | {pa[:, i].mean():.3e} | {ta[:, i].mean():.3e} |")
lines += ["", "truth aorta FWHM (s): " + str(rows[0].get("aorta_fwhm_truth_s", "n/a")), "seeds are paired (same ff_seed, same init seed per index). CS ref is a reference recon, not ground truth."]
open(f"{RES}/phantom_{a.sim}.md", "w").write("\n".join(lines)); print("\n".join(lines)); print("PHANTOM_EVAL_DONE")
