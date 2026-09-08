"""per-slice method-neutral NUFFT rulers for the kidney batch (A: pre-contrast ~240-spoke,
B: all-spoke static temporal-mean), same construction as nufft_reference.py (slice 13).
out: results_nufft_slice{Z}/{nufft_all,nufft_pre,pre_view_idx}.npy + meta.json"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, finufft, os, json
REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"; D = "/scratch/rnga/vvpshenov/DCE_NIK"
SLICES = [18, 19, 20, 21]
sh = np.load(f"{REF}/shared.npz")
traj = np.asarray(sh["traj_norm"]).astype(np.complex64); vt = np.asarray(sh["view_time"]).ravel().astype(np.float64)
TA = float(sh["TA"]); nx = int(sh["nx"]); bas = int(sh["bas"])

def ramp_dcf(tr): return np.maximum(np.abs(tr).astype(np.float64), (1.0 / nx) / 4.0)
def crop(img, n=bas): s = (img.shape[0] - n) // 2; return img[s:s + n, s:s + n]

for SL in SLICES:
    sl = np.load(f"{REF}/slice_{SL:02d}.npz"); kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64)
    b1 = np.asarray(sl["b1"]).astype(np.complex64); ncc = kdata.shape[2]; OUT = f"{D}/results_nufft_slice{SL}"; os.makedirs(OUT, exist_ok=True)
    def nufft_recon(view_idx, sign=-1.0, eps=1e-6):
        tr = traj[:, view_idx]; w = ramp_dcf(tr)
        x = (sign * 2 * np.pi * np.real(tr)).astype(np.float64).ravel(); y = (sign * 2 * np.pi * np.imag(tr)).astype(np.float64).ravel()
        ci = np.zeros((nx, nx, ncc), dtype=np.complex128)
        for c in range(ncc):
            ci[:, :, c] = finufft.nufft2d1(x, y, (kdata[:, view_idx, c] * w).astype(np.complex128).ravel(), (nx, nx), eps=eps, isign=1)
        return np.sum(ci * np.conj(b1), 2) / (np.sum(np.abs(b1) ** 2, 2) + 1e-12)
    cs_mean = np.abs(np.asarray(sl["cs_img"])).mean(-1); allv = np.arange(traj.shape[1]); best = None
    for sign in (-1.0, 1.0):
        im = np.abs(crop(nufft_recon(allv, sign=sign))); c = float(np.corrcoef(im.ravel(), cs_mean.ravel())[0, 1])
        if best is None or c > best[1]: best = (sign, c, im)
    SIGN, corr_all, img_all = best
    c0 = nx // 2; nav = np.sqrt((np.abs(kdata[c0]) ** 2).sum(-1)); o = np.argsort(vt); nav_s = nav[o]; t_s = vt[o] * TA
    k = 41; nav_sm = np.convolve(np.pad(nav_s, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    base = np.median(nav_sm[t_s < 30]); amp = nav_sm.max() - base; arrival = float(t_s[np.argmax(nav_sm > base + 0.15 * amp)])
    t_pre = max(20.0, arrival - 8.0); pre_idx = np.where(vt * TA < t_pre)[0]
    img_pre = np.abs(crop(nufft_recon(pre_idx, sign=SIGN)))
    np.save(f"{OUT}/nufft_all.npy", img_all.astype(np.float32)); np.save(f"{OUT}/nufft_pre.npy", img_pre.astype(np.float32)); np.save(f"{OUT}/pre_view_idx.npy", pre_idx)
    json.dump(dict(sign=SIGN, corr_all_vs_cs=corr_all, arrival_s=arrival, t_pre_s=t_pre, n_pre_spokes=int(len(pre_idx)), n_all_spokes=int(len(allv)), nx=nx, bas=bas), open(f"{OUT}/meta.json", "w"), indent=1)
    print(f"slice {SL}: sign {SIGN:+.0f} corr {corr_all:.4f} | arrival {arrival:.0f}s pre {len(pre_idx)} spokes -> {OUT}", flush=True)
print("RULERS DONE")
