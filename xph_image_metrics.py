"""broad image-quality panel vs XCAT ground truth, phantom slice ZI.

the four metrics used so far (haarpsi, ssim, psnr, nrmse) all reward smoothness, so a blurrier but
quieter recon can win them. this adds (a) more full-reference metrics, (b) a RESOLUTION vs NOISE
decomposition, and (c) spatial-frequency-band error, which says WHERE each method fails.
all methods scored on the same window-averaged truth and the same global LS scale.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, glob
import numpy as np, torch, piq
from masked_metrics import haarpsi_masked, ssim_masked
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import xph_pipeline as P, xph_common as X

A = f"{P.OUT}/arrays"; SW = f"{P.OUT}/v2_sweep"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = P.data(); tq = d["times"]; body = d["labels"] > 0
Tr = X.truth_at(P.ZI, tq); F = len(tq)
Rz = X.rois(P.ZI, d["labels"])

def winavg(v, G):
    n = v.shape[-1] // G
    return np.stack([v[:, :, g*G:(g+1)*G].mean(2) for g in range(n)], -1)

def prep(img, ref):
    """global LS scale, then per-frame [0,1] normalization by the truth 99.5 pct (same for both)."""
    s = np.sum(img[body]*ref[body]) / (np.sum(img[body]**2) + 1e-12)
    return img * s

def band_error(img, ref):
    """error split by spatial frequency: where does each method fail?"""
    out = {}
    R = np.fft.fftshift(np.fft.fft2(ref.mean(-1)))
    I = np.fft.fftshift(np.fft.fft2(img.mean(-1)))
    n = R.shape[0]; yy, xx = np.mgrid[:n, :n]; r = np.sqrt((yy-n/2)**2 + (xx-n/2)**2) / (n/2)
    for lab, lo, hi in (("low", 0, .15), ("mid", .15, .45), ("high", .45, 1.0)):
        m = (r >= lo) & (r < hi)
        out[f"err_{lab}k"] = float(np.linalg.norm(I[m]-R[m]) / (np.linalg.norm(R[m]) + 1e-30))
    return out

def sharpness_noise(img, ref):
    """resolution vs noise, separated. edge energy inside the body = resolution proxy;
    variance in a flat background region = noise proxy. reported RELATIVE to truth."""
    def edge(v):
        g = np.gradient(v.mean(-1)); return float(np.sqrt((g[0]**2 + g[1]**2)[body].mean()))
    bg = ~body
    def noi(v):
        x = v.mean(-1)[bg]; return float(x.std())
    return dict(sharpness_rel=edge(img)/(edge(ref)+1e-12), bg_noise=noi(img))

MT = None
def full_ref(img, ref):
    global MT
    if MT is None: MT = torch.from_numpy(body.astype(np.float32))[None, None].to(dev)
    out = {}
    nt = img.shape[-1]
    ts = range(0, nt, max(1, nt//40))
    acc = {k: [] for k in ("haarpsi","ssim","ms_ssim","gmsd","vif_p","dss","srsim","mdsi")}
    for t in ts:
        vmax = float(np.percentile(ref[:, :, t][body], 99.5)) + 1e-12
        a = torch.from_numpy(np.clip(img[:, :, t]/vmax, 0, 1)[None, None]).float().to(dev)
        b = torch.from_numpy(np.clip(ref[:, :, t]/vmax, 0, 1)[None, None]).float().to(dev)
        acc["haarpsi"].append(float(haarpsi_masked(a, b, MT, data_range=1.0).cpu()))   # body-masked
        acc["ssim"].append(float(ssim_masked(a, b, MT, data_range=1.0).cpu()))
        for k, fn in (("ms_ssim",piq.multi_scale_ssim),
                      ("gmsd",piq.gmsd),("vif_p",piq.vif_p),("dss",piq.dss),
                      ("srsim",piq.srsim),("mdsi",piq.mdsi)):
            try: acc[k].append(float(fn(a, b, data_range=1.0).cpu()))
            except Exception: pass
    for k, v in acc.items():
        if v: out[k] = float(np.mean(v))
    rv = float(ref[body].max()-ref[body].min())
    out["nrmse"] = float(np.mean([np.sqrt(np.mean((img[:,:,t][body]-ref[:,:,t][body])**2))/rv for t in range(nt)]))
    pk = float(ref[body].max())
    mse = float(np.mean([((img[:,:,t][body]-ref[:,:,t][body])**2).mean() for t in range(nt)]))
    out["psnr"] = 10*np.log10(pk**2/(mse+1e-20))
    return out

CASES = []
G_MATCH = 5   # 25 spokes/frame, the single setting. NIK window-averaged to grasp's 68 frames.
for n in ("sub16", "free"):
    p = f"{A}/nik_fine_{n}.npy"
    if os.path.exists(p): CASES.append((f"NIK-{n} (binned 68fr)", winavg(np.load(p), G_MATCH), G_MATCH))
for g, lam in ((5, "0.25"), (8, "0.02")):
    p = f"{SW}/v2_G{g:02d}.npy" if lam == "0.25" else f"{SW}/v2_G{g:02d}_lam{lam}.npy"
    if os.path.exists(p): CASES.append((f"GRASP-v2 {5*g}spf lam{lam}", np.load(p), g))

rows = []
for lab, img, G in CASES:
    ref = winavg(Tr, G) if G > 1 else Tr
    im = prep(np.abs(img).astype(np.float32), ref)
    r = dict(method=lab, frames=int(im.shape[-1]))
    r.update(full_ref(im, ref)); r.update(band_error(im, ref)); r.update(sharpness_noise(im, ref))
    rows.append(r); print(f"  scored {lab}", flush=True)

keys = ["haarpsi","ssim","ms_ssim","srsim","dss","vif_p","gmsd","mdsi","psnr","nrmse",
        "err_lowk","err_midk","err_highk","sharpness_rel","bg_noise"]
print(f"\n{'method':26} " + " ".join(f"{k[:8]:>8}" for k in keys))
for r in rows:
    print(f"{r['method']:26} " + " ".join(f"{r.get(k,float('nan')):>8.4f}" for k in keys))
print("\nhigher better: haarpsi ssim ms_ssim fsim srsim dss vif_p psnr sharpness_rel")
print("lower  better: gmsd mdsi nrmse err_*k bg_noise   (sharpness_rel: 1.0 = matches truth)")
json.dump(rows, open(f"{SW}/image_metrics_panel.json","w"), indent=1)
print("PANEL_DONE")
