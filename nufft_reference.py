"""Method-neutral NUFFT references for slice 13 (no rank cage, no learned prior).
  (A) ALL spokes      -> static reference for the TEMPORAL MEAN
  (B) pre-contrast    -> static reference for a window where the object truly is static
Density-compensated adjoint NUFFT per coil + SENSE combine with the SAME b1 as CS/NIK.
out: results_nufft/nufft_all.npy, nufft_pre.npy (+ meta)"""
import numpy as np, finufft, os, json
REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
OUT = "/scratch/rnga/vvpshenov/DCE_NIK/results_nufft"
os.makedirs(OUT, exist_ok=True)
SL = 13

sh = np.load(f"{REF}/shared.npz")
traj = np.asarray(sh["traj_norm"]).astype(np.complex64)      # [nx, ntviews], |k|<=0.5
vt = np.asarray(sh["view_time"]).ravel().astype(np.float64)  # 0..1
TA = float(sh["TA"]); nx = int(sh["nx"]); bas = int(sh["bas"])
sl = np.load(f"{REF}/slice_{SL:02d}.npz")
kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64)  # [nx, ntviews, ncc]
b1 = np.asarray(sl["b1"]).astype(np.complex64)               # [nx, nx, ncc]
ncc = kdata.shape[2]

def ramp_dcf(tr):
    """radial density compensation ~|k|, with a floor so DC is not annihilated."""
    r = np.abs(tr).astype(np.float64)
    dk = 1.0 / nx
    return np.maximum(r, dk / 4.0)

def nufft_recon(view_idx, sign=-1.0, eps=1e-6):
    """density-compensated adjoint NUFFT over the given spokes -> SENSE image [nx,nx]."""
    tr = traj[:, view_idx]
    w = ramp_dcf(tr)
    # finufft type 1: coords in [-pi,pi]. sign flips the frame (grasp vs finufft convention).
    x = (sign * 2.0 * np.pi * np.real(tr)).astype(np.float64).ravel()
    y = (sign * 2.0 * np.pi * np.imag(tr)).astype(np.float64).ravel()
    coil_imgs = np.zeros((nx, nx, ncc), dtype=np.complex128)
    for c in range(ncc):
        cvals = (kdata[:, view_idx, c] * w).astype(np.complex128).ravel()
        coil_imgs[:, :, c] = finufft.nufft2d1(x, y, cvals, (nx, nx), eps=eps, isign=1)
    num = np.sum(coil_imgs * np.conj(b1), axis=2)
    den = np.sum(np.abs(b1) ** 2, axis=2) + 1e-12
    return num / den

def crop(img, n=bas):
    s = (img.shape[0] - n) // 2
    return img[s:s + n, s:s + n]

# ---- validate the frame/sign convention against the CS temporal mean ----
cs_mean = np.abs(np.asarray(sl["cs_img"])).mean(-1)              # [bas,bas]
allv = np.arange(traj.shape[1])
best = None
for sign in (-1.0, +1.0):
    im = np.abs(crop(nufft_recon(allv, sign=sign)))
    c = float(np.corrcoef(im.ravel(), cs_mean.ravel())[0, 1])
    print(f"  sign {sign:+.0f}: corr vs CS mean = {c:.4f}", flush=True)
    if best is None or c > best[1]:
        best = (sign, c, im)
SIGN, corr_all, img_all = best
print(f"-> using sign {SIGN:+.0f} (corr {corr_all:.4f})", flush=True)

# ---- pre-contrast window: find bolus arrival from the k=0 navigator ----
c0 = nx // 2
nav = np.sqrt((np.abs(kdata[c0]) ** 2).sum(-1))                  # [ntviews]
o = np.argsort(vt); nav_s = nav[o]; t_s = vt[o] * TA
k = 41
nav_sm = np.convolve(np.pad(nav_s, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
base = np.median(nav_sm[t_s < 30]); amp = nav_sm.max() - base
arrival = float(t_s[np.argmax(nav_sm > base + 0.15 * amp)])      # 15% of rise
t_pre = max(20.0, arrival - 8.0)                                 # safety margin
pre_idx = np.where(vt * TA < t_pre)[0]
print(f"bolus arrival ~{arrival:.0f}s -> pre-contrast window t<{t_pre:.0f}s = {len(pre_idx)} spokes", flush=True)

img_pre = np.abs(crop(nufft_recon(pre_idx, sign=SIGN)))
np.save(f"{OUT}/nufft_all.npy", img_all.astype(np.float32))
np.save(f"{OUT}/nufft_pre.npy", img_pre.astype(np.float32))
np.save(f"{OUT}/pre_view_idx.npy", pre_idx)
json.dump(dict(sign=SIGN, corr_all_vs_cs=corr_all, arrival_s=arrival, t_pre_s=t_pre,
               n_pre_spokes=int(len(pre_idx)), n_all_spokes=int(len(allv)), nx=nx, bas=bas),
          open(f"{OUT}/meta.json", "w"), indent=1)
print(f"wrote {OUT}/nufft_all.npy {img_all.shape}, nufft_pre.npy {img_pre.shape}")
