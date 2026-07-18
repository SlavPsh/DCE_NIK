"""Port verification: compare DCE_NIK train_grasp_nik (winning recipe) against the
autoresearch numbers on the SAME slice-13 data.
target (e8_ts1p5_tf32): held-out ~0.323, swing ~51%, nav-corr ~0.885."""
import sys, re, glob, numpy as np
D = "/scratch/rnga/vvpshenov/DCE_NIK"
# argv: [recon_dir] [log_glob]   defaults = the port-verify run
recon_dir = sys.argv[1] if len(sys.argv) > 1 else f"{D}/results_nik_verify"
log_glob = sys.argv[2] if len(sys.argv) > 2 else f"{D}/slurm-nik-verify-*.out"
recon = np.abs(np.load(f"{recon_dir}/nik_slice_13.npy"))               # [bas,bas,nt]
nt = recon.shape[-1]

# held-out from the training log
best = None
for lg in sorted(glob.glob(log_glob)):
    for line in open(lg):
        m = re.search(r"restored best \(heldout ([\d.eE+-]+)\)", line)
        if m: best = float(m.group(1))
print(f"[analysis] {recon_dir.split('/')[-1]}  held-out MSE = {best}   (full-rank baseline 0.3245)")

# navigator (k=0 self-nav) + ROI, from the same slice-13 raw data
sl = np.load("/scratch/rnga/vvpshenov/grasp_pro_py/results_ref/slice_13.npz")
krad = np.asarray(sl["kdata_radial"]); cs = np.abs(np.asarray(sl["cs_img"]))
# view_time / trajectory from shared
sh = np.load("/scratch/rnga/vvpshenov/grasp_pro_py/results_ref/shared.npz")
vt = np.asarray(sh["view_time"]).ravel()
c0 = krad.shape[0] // 2
dc = np.sqrt((np.abs(krad[c0, :, :]) ** 2).sum(-1))                     # |k=0| per spoke
o = np.argsort(vt); ts = vt[o]; dcs = np.clip(dc[o], np.percentile(dc, 1), np.percentile(dc, 99))
nav = np.convolve(np.pad(dcs, (30, 30), mode="reflect"), np.ones(61)/61, mode="valid")[:len(dcs)]
nav = nav / np.median(nav[int(.35*len(nav)):])

def n01(a): a = np.asarray(a, float); return (a - a.min()) / (a.max() - a.min() + 1e-12)
roi = recon.mean(-1) > np.quantile(recon.mean(-1), 0.6)
curve = np.array([recon[..., i][roi].mean() for i in range(nt)])
swing = (curve.max() - curve.min()) / curve.mean() * 100
tt = np.linspace(0, 1, nt)
navcorr = float(np.corrcoef(n01(curve), np.interp(tt, ts, n01(nav)))[0, 1])
print(f"[verify] recon @ {nt} frames:  swing {swing:.1f}%   nav-corr {navcorr:+.3f}")
print(f"[verify] targets: swing ~51%   nav-corr ~0.885")
# scale-matched agreement vs CS reference (sanity the image is sensible)
a, b = recon.mean(-1).ravel(), cs.mean(-1).ravel() if cs.ndim == 3 else cs.ravel()
if a.shape == b.shape:
    print(f"[verify] recon vs CS static corr = {np.corrcoef(a, b)[0,1]:.3f}")
