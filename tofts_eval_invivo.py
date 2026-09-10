"""step 5 in-vivo eval (slices 18/19/21), identical pipeline for both arms. no ground truth: model-free NUFFT curves
(raw + affine), physical bounds, held-out VAL/TEST spoke complex k-space NMSE by annulus (NIK arms only; CS files are
magnitude -> held-out blocked for CS). CS rows (GRASP-v2 f25, GRASP-Pro f25; 122 fr) are references, never truth."""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, glob, re
import numpy as np, torch
from types import SimpleNamespace
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import consolidated as C, nik_adapter as A
from kspace_normalization import compute_dcf_radial, KSpaceNormalizer
from train_grasp_nik import build_model
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = "/net/beegfs/users/P101440/DCE_NIK"; RES = f"{B}/results/tofts_vs_patlak"; IV = f"{RES}/invivo"; REFD = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"
GV = "/net/beegfs/users/P101440/grasp_v2/results_grasp_v2"; GP = "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs"
TA = 375.0; NTV = 1710; ROIS = ("aorta", "cortex", "medulla", "liver"); EDGES = np.linspace(0, 1, 17)
KEEP = np.load(f"{B}/spoke_masks/keep_f25.npy"); VAL = np.load(f"{B}/spoke_masks/val_f25c_m8.npy"); TEST = np.load(f"{B}/spoke_masks/test_f25c_m9.npy")
sh = A.load_shared(REFD)
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--slices", default="18,19,21"); _ap.add_argument("--arms", default="patlak,tofts")
_ap.add_argument("--suffix", default=None, help="outputs invivo<suffix>.{json,md}; default '' for f25, '_k80' for k80")
_ap.add_argument("--spokes", default="f25", choices=["f25", "k80"], help="k80 = the standard: keep v%10<8 (1368 views), val v%10==8, test v%10==9, runs in invivo_k80/, grasp k80 refs"); _a = _ap.parse_args()
SLICES = [int(z) for z in _a.slices.split(",")]; ARMS = _a.arms.split(",")
if _a.spokes == "k80":
    KEEP = np.load(f"{B}/spoke_masks/keep_f80match.npy"); VAL = np.load(f"{B}/spoke_masks/val_k80_m8.npy"); TEST = np.load(f"{B}/spoke_masks/test_k80_m9.npy"); IV = f"{RES}/invivo_k80"
    REFS = (("GRASP-v2 k80 (1368 views, 142 fr, n12 lam0.25)", "{GV}/gv2_slice{Z}_n12_k80.npy"), ("GRASP-Pro f80match (1368 views, 122 fr, K5)", "{GP}/cs_slice{Z}_f80match.npy"))
    SUF = "_k80" if _a.suffix is None else _a.suffix; SPK = "k80 = 1368/1708 views (v%10<8), VAL v%10==8 for early stop, TEST v%10==9 untouched; same views for every method"
else:
    REFS = (("GRASP-v2 f25 (488 spokes, 122 fr, lam0.25)", "{GV}/gv2_slice{Z}_f25.npy"), ("GRASP-Pro f25 (488 spokes, 122 fr, K5)", "{GP}/cs_slice{Z}_f25.npy"))
    SUF = _a.suffix or ""; SPK = "keep_f25 = 488/1710 spokes, VAL v%10==8 of complement for early stop, TEST v%10==9 untouched"
LABEL = {"patlak": "Patlak", "tofts": "Tofts"}
def basis_for(arm, Z):                                                          # tofts = rank rule basis, tofts<R> = forced rank R
    return f"{RES}/basis_sl{Z}_r{arm[5:]}.npz" if arm.startswith("tofts") and arm != "tofts" else f"{RES}/basis_sl{Z}.npz"

def bs(c): return c - np.median(c[:8])
def ft(nt): e = np.linspace(0, TA, nt+1); return 0.5*(e[:-1]+e[1:])
def fwhm(t, c):
    c = bs(c); i = int(np.argmax(c)); h = c[i]/2; l = i; r = i
    while l > 0 and c[l] > h: l -= 1
    while r < len(c)-1 and c[r] > h: r += 1
    neg = float(np.mean(c < -0.05*c[i])); mono = float(np.mean(np.diff(c[:i+1]) >= -0.02*c[i])) if i > 1 else 1.0
    return dict(aorta_fwhm_s=float(t[r]-t[l]), aorta_ttp_s=float(t[i]), aorta_neg_frac=neg, aorta_rise_mono=mono)
def refwin(mf, tmf, nt):
    """model-free [240,H,W] window-averaged onto nt frames -> [H,W,nt] (same global LS scale rule as final_single_setting)"""
    e = np.linspace(0, TA, nt+1); o = np.zeros((mf.shape[1], mf.shape[2], nt), np.float32)
    for g in range(nt):
        m = (tmf >= e[g]) & (tmf < e[g+1])
        if not m.any(): m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf-0.5*(e[g]+e[g+1])))] = True
        o[:, :, g] = mf[m].mean(0)
    return o
def curves(v, t, ctx):
    tmf = ctx["tmf"]; mf = ctx["mf"]; rois = ctx["rois"]; out = {}
    R = refwin(mf, tmf, v.shape[-1]); bd = ctx["BODY"]; v = v * (np.sum(v[bd]*R[bd]) / (np.sum(v[bd]**2) + 1e-12))   # ONE global scale vs model-free
    for roi in ROIS:
        if roi not in rois or rois[roi].sum() == 0: continue
        c = np.interp(tmf, t, np.array([v[..., i][rois[roi]].mean() for i in range(v.shape[-1])])); m = np.array([im[rois[roi]].mean() for im in mf])
        cb, mb = bs(c), bs(m); s = float((cb @ mb)/(cb @ cb + 1e-12))                                  # global-ish: one scale per ROI, offset by baseline
        out[f"mf_{roi}_scale"] = float(np.linalg.norm(s*cb-mb)/np.linalg.norm(mb))
        Aa = np.stack([c, np.ones_like(c)], 1); ab = np.linalg.lstsq(Aa, m, rcond=None)[0]                  # affine (raw + affine), the corrected temporal ruler
        out[f"mf_{roi}_affine"] = float(np.linalg.norm(Aa @ ab - m)/np.linalg.norm(m - m.mean() + m.mean()))
        if roi == "aorta":
            out["aorta_peak_ratio_vs_mf"] = float(cb.max()/mb.max()); out.update(fwhm(tmf, c)); mfw = fwhm(tmf, m); out["mf_aorta_fwhm_s"] = mfw["aorta_fwhm_s"]
    # cortex vs medulla separation preserved? (late-phase correlation, physiology says distinct)
    if "cortex" in rois and "medulla" in rois:
        tt = t; cc = np.array([v[..., i][rois["cortex"]].mean() for i in range(v.shape[-1])]); mm = np.array([v[..., i][rois["medulla"]].mean() for i in range(v.shape[-1])])
        lt = tt > 120; out["cortex_medulla_late_corr"] = float(np.corrcoef(bs(cc)[lt], bs(mm)[lt])[0, 1])
    return out

@torch.no_grad()
def heldout(run, Z, arm="tofts"):
    ck = torch.load(f"{run}/model_slice_{Z:02d}.pt", map_location=dev, weights_only=False)
    ds = A.make_radial_dataset(REFD, Z, compute_device=dev, shared=sh); x, t, c, y_raw, sid = ds["x_all"], ds["t_all"], ds["coil_all"], ds["y_all_raw"], ds["spoke_id_all"]
    kept = torch.as_tensor(KEEP, device=dev, dtype=sid.dtype); tr = torch.where(torch.isin(sid, kept))[0]
    dcf = compute_dcf_radial(x, method="simple_ramp"); nz = KSpaceNormalizer(); nz.fit(x[tr], y_raw[tr], dcf=dcf[tr], envelope_exponent=0.75); y = nz.normalize(x, y_raw)
    args = SimpleNamespace(**{k: ck[k] for k in ("model", "rank", "hidden", "depth", "w0", "s0", "coil_embed_dim", "k_freq", "k_sigma", "t_freq", "t_sigma", "ff_seed")},
                           patlak_free=0, aif_file=f"{B}/aif_slice{Z}.npz", tofts_basis=basis_for(arm, Z), phi_hidden=64, phi_depth=3, phi_w0=30.0, phi_ortho=False, n_pk=-1, radial_alpha=1.0)
    m = build_model(args, int(ck["ncc"])).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
    out = dict(rank=int(m.rank), params=int(sum(p.numel() for p in m.parameters())))
    for nm, spk in (("val", VAL), ("test", TEST), ("train", KEEP)):
        idx = torch.where(torch.isin(sid, torch.as_tensor(spk, device=dev, dtype=sid.dtype)))[0]; num = np.zeros(16); den = np.zeros(16)
        for i in range(0, idx.numel(), 40000):
            j = idx[i:i+40000]; pr = nz.denormalize(x[j], m(x[j], t[j], c[j])); yt = nz.denormalize(x[j], y[j])
            e = (np.abs((pr[:, 0]+1j*pr[:, 1]).cpu().numpy() - (yt[:, 0]+1j*yt[:, 1]).cpu().numpy()))**2; p = np.abs((yt[:, 0]+1j*yt[:, 1]).cpu().numpy())**2
            r = torch.sqrt(x[j, 0]**2 + x[j, 1]**2).cpu().numpy(); b = np.clip(np.digitize(r, EDGES)-1, 0, 15)   # model coords in [-1,1] (traj_norm*2), r in [0,1]
            for k in range(16): mk = b == k; num[k] += e[mk].sum(); den[k] += p[mk].sum()
        out[f"{nm}_kNMSE"] = float(num.sum()/den.sum()); out[f"{nm}_annuli"] = (num/(den+1e-30)).tolist()
    del m; torch.cuda.empty_cache(); return out

def resources(Z, arm, s):
    """wall/peak mem from the slurm log (invivo_<array>_<i>.log, i = slice_idx*6 + model_idx*3 + seed)"""
    wj = f"{IV}/{arm}_sl{Z}_s{s}/wandb_runs/slice_{Z:02d}.json"                  # nik_wandb summary, newer runs
    if os.path.exists(wj):
        j = json.load(open(wj)); return dict(wall_s=float(j.get("wall_s", np.nan)), peak_gpu_mb=float(j.get("peak_gpu_mb", np.nan)))
    i = {18: 0, 19: 1, 21: 2}[Z]*6 + (0 if arm == "patlak" else 1)*3 + s; out = dict(wall_s=np.nan, peak_gpu_mb=np.nan)
    for f in glob.glob(f"{RES}/logs/invivo_*_{i}.log"):
        s_ = open(f).read(); m1 = re.search(r"\((\d+)s\)", s_); m2 = re.search(r"peak_gpu_MB (\d+)", s_)
        if m1: out["wall_s"] = float(m1.group(1))
        if m2: out["peak_gpu_mb"] = float(m2.group(1))
    return out

rows = []
for Z in SLICES:
    ctx = C.slice_ctx(Z); md = np.load(f"{B}/step2_slice{Z}.npz"); ctx["mf"] = md["mf"]; ctx["tmf"] = md["tmf"]   # model-free frames-first [240,192,192]
    for arm in ARMS:
        for s in (0, 1, 2):
            run = f"{IV}/{arm}_sl{Z}_s{s}"; f = f"{run}/nik_slice_{Z:02d}_cplx.npy"
            if not os.path.exists(f): print(f"  sl{Z} {arm} s{s}: MISSING"); rows.append(dict(slice=Z, arm=arm, seed=s, status="missing")); continue
            v = np.abs(np.load(f)).astype(np.float32); r = dict(slice=Z, arm=arm, seed=s, status="complete", frames=int(v.shape[-1]))
            r.update(curves(v, ft(v.shape[-1]), ctx))
            v122 = np.stack([np.interp(ft(122), ft(v.shape[-1]), v.reshape(-1, v.shape[-1])[i]) for i in range(v.shape[0]*v.shape[1])], 0).reshape(v.shape[0], v.shape[1], 122) if False else None
            try: r.update(heldout(run, Z, arm))
            except Exception as e: r["heldout_error"] = str(e)[:120]; print("   heldout failed:", str(e)[:120])
            r.update(resources(Z, arm, s)); rows.append(r)
            print(f"  sl{Z} {arm} s{s}: aorta_aff {r.get('mf_aorta_affine', np.nan):.4f} cortex_aff {r.get('mf_cortex_affine', np.nan):.4f} fwhm {r.get('aorta_fwhm_s', np.nan):.1f}s | val {r.get('val_kNMSE', np.nan):.3e} test {r.get('test_kNMSE', np.nan):.3e}", flush=True)
    for lab, pt in REFS:
        p = pt.format(GV=GV, GP=GP, Z=Z)
        if not os.path.exists(p): continue
        v = np.abs(np.load(p)).astype(np.float32); r = dict(slice=Z, arm=lab, seed=-1, status="complete", frames=int(v.shape[-1])); r.update(curves(v, ft(v.shape[-1]), ctx)); rows.append(r)
json.dump(rows, open(f"{RES}/invivo{SUF}.json", "w"), indent=1)
keys = ["mf_aorta_affine", "mf_cortex_affine", "mf_medulla_affine", "mf_liver_affine", "mf_aorta_scale", "mf_cortex_scale", "mf_medulla_scale", "aorta_peak_ratio_vs_mf",
        "aorta_fwhm_s", "aorta_ttp_s", "aorta_neg_frac", "aorta_rise_mono", "cortex_medulla_late_corr", "train_kNMSE", "val_kNMSE", "test_kNMSE", "wall_s", "peak_gpu_mb", "params"]
lines = [f"# in vivo (meas_p3_dce, slices {'/'.join(map(str, SLICES))}, {SPK})", "",
         "rulers: mf_* = NRMSE vs model-free NUFFT ROI curve on its 240-pt grid (affine = raw+affine fit; scale = baseline-subtracted single scale). physical bounds on aorta. *_kNMSE = complex k-space NMSE at held-out spokes (NIK only). CS rows are references, NOT truth; CS held-out blocked (magnitude-only files).", ""]
L = [LABEL.get(a, a) for a in ARMS]
for Z in SLICES:
    lines += [f"## slice {Z}", "| metric | " + " | ".join(f"{l} mean±SD (n)" for l in L) + " | " + " | ".join(f"Δ {l}−{L[0]}" for l in L[1:]) + " | GRASP-v2 f25 | GRASP-Pro f25 |",
              "|" + "---|" * (2 * len(ARMS) + 2)]
    def ms(arm, k):
        v = np.array([r[k] for r in rows if r.get("slice") == Z and r.get("arm") == arm and r.get("status") == "complete" and k in r], float); return (np.nanmean(v), np.nanstd(v), v.size) if v.size else (np.nan, np.nan, 0)
    def cs(pre, k):
        v = [r.get(k, np.nan) for r in rows if r.get("slice") == Z and str(r.get("arm", "")).startswith(pre)]; return v[0] if v else np.nan
    for k in keys:
        M = [ms(a, k) for a in ARMS]
        lines.append(f"| {k} | " + " | ".join(f"{m[0]:.4g} ± {m[1]:.2g} ({m[2]})" for m in M) + " | " + " | ".join(f"{m[0]-M[0][0]:+.4g}" for m in M[1:])
                     + f" | {cs('GRASP-v2', k):.4g} | {cs('GRASP-Pro', k):.4g} |")
    mfw = [r.get("mf_aorta_fwhm_s") for r in rows if r.get("slice") == Z and "mf_aorta_fwhm_s" in r]; lines.append(f"\nmodel-free aorta FWHM (s): {mfw[0] if mfw else 'n/a'}\n")
    AN = [np.array([r["test_annuli"] for r in rows if r.get("slice") == Z and r.get("arm") == a and "test_annuli" in r]) for a in ARMS]
    if all(x.size for x in AN):
        lines += (["TEST-spoke k-space NMSE per |k| annulus:", "| annulus | " + " | ".join(L) + " |", "|" + "---|" * (len(ARMS) + 1)]
                  + [f"| {EDGES[i]:.2f}-{EDGES[i+1]:.2f} | " + " | ".join(f"{x[:, i].mean():.3e}" for x in AN) + " |" for i in range(16)] + [""])
open(f"{RES}/invivo{SUF}.md", "w").write("\n".join(lines)); print("\n".join(lines)); print("INVIVO_EVAL_DONE")
