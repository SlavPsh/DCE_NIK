"""CONSOLIDATED batch analysis + report (P1 replication, P2 rank pareto, P3 Patlak).
applies the standing metric guards: corrected spatial eval (2 rulers, signed bgE, diff-
structure) at f25; realized rank = SVD of the COMPLEX recon; ROI-averaged (not per-voxel)
temporal curves vs streak-free model-free; oscillation reported with every divergence.
missing/failed configs are logged and skipped, never abort the batch.
usage: python consolidated.py    (writes figures/ + prints the report + report_data.json)"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.ndimage as ndi, json, os, sys, torch
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
from figpath import fig as fpath
from masked_metrics import haarpsi_masked, ssim_masked
D = "/scratch/rnga/vvpshenov/DCE_NIK"; REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
# reference-method plumbing. defaults = grasp-pro (unchanged). CSD/CSPRE swap in grasp v2.
# NOTE: ROI anatomy stays on grasp-pro cs_img on purpose, so both notebooks score the SAME rois.
# only the reference-METHOD image (ctx["cs_meth"]) follows CSPRE.
CSD = os.environ.get("CSD", "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs")
CSPRE = os.environ.get("CSPRE", "cs")
BATCH = f"{D}/results_batch"; TA = 375.0
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---- config matrix ----
CFG = []
for Z in (18, 19, 20):
    CFG += [(Z, "R16", f"{BATCH}/nik_r16_sl{Z}/nik_slice_{Z}_cplx.npy", "sub", 16),
            (Z, "full", f"{BATCH}/full_sl{Z}/nik_slice_{Z}_cplx.npy", "res", None)]
CFG += [(21, "R8", f"{BATCH}/nik_r8_sl21/nik_slice_21_cplx.npy", "sub", 8),
        (21, "R16", f"{BATCH}/nik_r16_sl21/nik_slice_21_cplx.npy", "sub", 16),
        (21, "R32", f"{BATCH}/nik_r32_sl21/nik_slice_21_cplx.npy", "sub", 32),
        (21, "R64", f"{BATCH}/nik_r64_sl21/nik_slice_21_cplx.npy", "sub", 64),
        (21, "full", f"{BATCH}/full_sl21f25/nik_slice_21_cplx.npy", "res", None),
        (21, "full_f100", f"{D}/results_spoke_full_slice21/nik_slice_21_cplx.npy", "res", None),
        (21, "PK_F0", f"{BATCH}/pk_f0_sl21/nik_slice_21_cplx.npy", "pk", 3),
        (21, "PK_F2", f"{BATCH}/pk_f2_sl21/nik_slice_21_cplx.npy", "pk", 5),
        (21, "PK_F4", f"{BATCH}/pk_f4_sl21/nik_slice_21_cplx.npy", "pk", 7)]

# ---- per-slice rulers, masks, ROIs (cached) ----
_slice_cache = {}
def slice_ctx(Z):
    if Z in _slice_cache: return _slice_cache[Z]
    NUF = f"{D}/results_nufft_slice{Z}"
    ref_pre = np.load(f"{NUF}/nufft_pre.npy").astype(np.float32); ref_all = np.load(f"{NUF}/nufft_all.npy").astype(np.float32)
    meta = json.load(open(f"{NUF}/meta.json")); t_pre = meta["t_pre_s"]
    def clean_body(ref):
        m0 = ndi.binary_opening(ref > np.quantile(ref, 0.55), iterations=2); l, n = ndi.label(m0)
        if n: m0 = (l == 1 + int(np.argmax(ndi.sum(np.ones_like(l), l, range(1, n + 1)))))
        return ndi.binary_closing(ndi.binary_fill_holes(m0), iterations=2)
    BODY = clean_body(ref_all); AIR = ~ndi.binary_dilation(BODY, iterations=4)
    cs100 = np.abs(np.asarray(np.load(f"{REF}/slice_{Z:02d}.npz")["cs_img"])).astype(np.float32)   # roi anatomy, always grasp-pro
    if CSPRE == "cs":
        cs_meth = cs100
    else:                                            # matched binning: p05 = NLINE 5 -> 342 frames
        _p = f"{CSD}/{CSPRE}_slice{Z:02d}_p05.npy"
        cs_meth = np.abs(np.load(_p)).astype(np.float32) if os.path.exists(_p) else None
    cv = cs100.std(-1) / (cs100.mean(-1) + 1e-6); NONENH = BODY & (cv < np.quantile(cv[BODY], 0.5))
    # ROIs from cs anatomy
    tC = np.linspace(0, TA, cs100.shape[-1]); base = cs100[..., tC < 50].mean(-1); enh = cs100 - base[..., None]
    late = enh[..., (tC > 130) & (tC < 200)].mean(-1); kid = BODY & (late > np.quantile(late[BODY], 0.985)); kid = ndi.binary_opening(kid, iterations=1)
    l, k = ndi.label(kid); kid = (l == (1 + np.argmax(ndi.sum(np.ones_like(l), l, range(1, k + 1))))) if k else kid
    fcm = int(np.argmin(np.abs(tC - 85))); thr = np.median(enh[..., fcm][kid]); cortex = kid & (enh[..., fcm] > thr); medulla = kid & (enh[..., fcm] <= thr)
    early = enh[..., (tC > 40) & (tC < 80)].mean(-1); ao = BODY & (early > np.quantile(early[BODY], 0.995)) & (~kid); ao = ndi.binary_opening(ao, iterations=1)
    la, ka = ndi.label(ao); ao = (la == (1 + np.argmax(ndi.sum(np.ones_like(la), la, range(1, ka + 1))))) if ka else ao
    liver = ndi.binary_erosion(BODY, iterations=8) & (cv < np.quantile(cv[BODY], 0.3)) & (~kid)  # static interior
    ll, lk = ndi.label(liver); liver = (ll == (1 + np.argmax(ndi.sum(np.ones_like(ll), ll, range(1, lk + 1))))) if lk else liver
    # streak-free real ROI curves from model-free
    md = np.load(f"{D}/step2_slice{Z}.npz"); mf = md["mf"]; tmf = md["tmf"]
    def norm(t, c): b = c[t < 50].mean(); pk = c[(t > 20) & (t < 210)].max(); return (c - b) / (pk - b + 1e-9)
    real = {}
    for nm, mask in [("cortex", cortex), ("medulla", medulla), ("aorta", ao), ("liver", liver)]:
        if int(mask.sum()) == 0: continue
        rc = norm(tmf, np.array([im[mask].mean() for im in mf]))
        if np.isfinite(rc).all(): real[nm] = rc
    ctx = dict(ref_pre=ref_pre, ref_all=ref_all, t_pre=t_pre, BODY=BODY, AIR=AIR, NONENH=NONENH,
               rois=dict(cortex=cortex, medulla=medulla, aorta=ao, liver=liver), real=real, tmf=tmf, fcm=fcm, cs100=cs100, cs_meth=cs_meth)
    _slice_cache[Z] = ctx; return ctx

def norm(t, c): b = c[t < 50].mean(); pk = c[(t > 20) & (t < 210)].max(); return (c - b) / (pk - b + 1e-9)
def osc(t, n): sm = savgol_filter(n, 11, 3); return float(np.std(n - sm) / (np.abs(sm).max() + 1e-9))
def plateau(t, n): lt = (t > 100) & (t < 210); return float(np.median(n[lt]))

# ---- corrected spatial eval ----
def _ls(pred, ref, m): return float((pred[m] @ ref[m]) / (pred[m] @ pred[m] + 1e-12))
def _tb(img, vmax): return torch.from_numpy(np.clip(img / (vmax + 1e-12), 0, 1)[None, None]).float().to(dev)
def score(pred, ref, mask):
    s = _ls(pred, ref, mask); p = pred * s; mse = float(((p[mask] - ref[mask]) ** 2).mean()); pk = float(ref[mask].max())
    psnr = 10 * np.log10(pk ** 2 / (mse + 1e-20)); vmax = float(np.percentile(ref[mask], 99.5))
    mt = torch.from_numpy(mask.astype(np.float32))[None, None].to(dev)
    with torch.no_grad():
        h = float(haarpsi_masked(_tb(p, vmax), _tb(ref, vmax), mt, data_range=1.0).cpu())
        ss = float(ssim_masked(_tb(p, vmax), _tb(ref, vmax), mt, data_range=1.0).cpu())
    return dict(psnr=psnr, haarpsi=h, ssim=ss)
def bgE(pred, ref, BODY, AIR): s = _ls(pred, ref, BODY); return float(np.sqrt(((pred * s)[AIR] ** 2).mean()) / (np.sqrt((ref[BODY] ** 2).mean()) + 1e-12))
def ref_bgE(ref, BODY, AIR): return float(np.sqrt((ref[AIR] ** 2).mean()) / (np.sqrt((ref[BODY] ** 2).mean()) + 1e-12))
def diff_structure(pred, ref, BODY, AIR):
    s = _ls(pred, ref, BODY); d = (pred * s - ref) ** 2
    gg = ndi.gaussian_gradient_magnitude(ref, 1.0); edges = BODY & (gg > np.quantile(gg[BODY], 0.8))
    return dict(outside=float(d[AIR].sum() / (d.sum() + 1e-20)), inside_edge=float(d[edges].sum() / (d[BODY].sum() + 1e-20)))

def spatial_eval(pred_dyn, Z):
    ctx = slice_ctx(Z); t = np.linspace(0, TA, pred_dyn.shape[-1])
    pre = pred_dyn[..., t < ctx["t_pre"]].mean(-1); full = pred_dyn.mean(-1)
    A = score(pre, ctx["ref_pre"], ctx["BODY"]); B = score(full, ctx["ref_all"], ctx["NONENH"])
    return dict(rulerA_haarpsi=A["haarpsi"], rulerA_psnr=A["psnr"], rulerB_haarpsi=B["haarpsi"],
                bgE=bgE(full, ctx["ref_all"], ctx["BODY"], ctx["AIR"]), ref_bgE=ref_bgE(ctx["ref_all"], ctx["BODY"], ctx["AIR"]),
                **{f"diff_{k}": v for k, v in diff_structure(full, ctx["ref_all"], ctx["BODY"], ctx["AIR"]).items()})

def realized_rank(cplx, BODY):
    X = cplx[BODY]; Xc = X - X.mean(1, keepdims=True); s = np.linalg.svd(Xc, compute_uv=False, full_matrices=False)
    ev = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(ev, 0.99) + 1), int(np.searchsorted(ev, 0.999) + 1)

def temporal_eval(pred_dyn, Z):
    ctx = slice_ctx(Z); t = np.linspace(0, TA, pred_dyn.shape[-1]); out = {}
    for nm, mask in ctx["rois"].items():
        if int(mask.sum()) == 0 or nm not in ctx["real"]: continue     # skip empty/undefined ROI (e.g. sl20 aorta)
        c = norm(t, np.array([pred_dyn[..., i][mask].mean() for i in range(pred_dyn.shape[-1])]))
        if not np.isfinite(c).all(): continue
        rc = ctx["real"][nm]; nr = float(np.sqrt(np.mean((c - np.interp(t, ctx["tmf"], rc)) ** 2)))
        out[nm] = dict(plateau=plateau(t, c), osc=osc(t, c), nrmse=nr)
    return out

def load_cs(Z):  # CS f25, 122 frames
    p = f"{CSD}/{CSPRE}_slice{Z:02d}_f25.npy"
    return np.abs(np.load(p)).astype(np.float32) if os.path.exists(p) else None

if __name__ == "__main__":
    rows = []; missing = []
    # CS baseline per slice
    for Z in (18, 19, 20, 21):
        cs = load_cs(Z)
        if cs is None: missing.append(f"CS f25 slice {Z}"); continue
        r = dict(slice=Z, cfg="CS", model="cs", rrank=5)
        try: r["spatial"] = spatial_eval(cs, Z); r["temporal"] = temporal_eval(cs, Z)
        except Exception as e: r["error"] = str(e)
        rows.append(r)
    for (Z, lab, path, model, nom) in CFG:
        if not os.path.exists(path): missing.append(f"{lab} slice {Z} ({path.split('/')[-2]})"); continue
        cplx = np.load(path).astype(np.complex64); mag = np.abs(cplx)
        ctx = slice_ctx(Z); r = dict(slice=Z, cfg=lab, model=model, nom_rank=nom)
        try:
            r99, r999 = realized_rank(cplx, ctx["BODY"]); r["rrank"] = r99; r["rrank999"] = r999
            r["spatial"] = spatial_eval(mag, Z); r["temporal"] = temporal_eval(mag, Z)
        except Exception as e: r["error"] = str(e)
        rows.append(r)
    json.dump(rows, open(f"{D}/report_data.json", "w"), indent=1, default=float)
    print(f"analyzed {len(rows)} configs, {len(missing)} missing")
    for m in missing: print("  MISSING:", m)
    # compact matrix print
    print(f"\n{'slice':>5} {'cfg':>10} {'rrank':>6} {'A_haar':>7} {'B_haar':>7} {'bgE':>6} {'refbg':>6} {'cortex_nrmse':>12} {'cortex_plat':>11} {'cortex_osc':>10}")
    for r in rows:
        if "spatial" not in r or "temporal" not in r: print(f"{r['slice']:>5} {r['cfg']:>10}  ERROR {r.get('error','?')[:50]}"); continue
        s = r["spatial"]; tc = r["temporal"]["cortex"]
        print(f"{r['slice']:>5} {r['cfg']:>10} {r.get('rrank','?'):>6} {s['rulerA_haarpsi']:>7.3f} {s['rulerB_haarpsi']:>7.3f} {s['bgE']:>6.3f} {s['ref_bgE']:>6.3f} {tc['nrmse']:>12.3f} {tc['plateau']:>11.3f} {tc['osc']:>10.3f}")
