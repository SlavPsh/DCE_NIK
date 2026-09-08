"""Aggregate the |k|-dependent-FF (radial warp) diagnostic: for each alpha, held-out (log)
+ high-freq power ratio (blur proxy) + swing. The question: does the warp raise HF energy
(reduce blur) or just move held-out / add grain? Baseline: full-rank NIK HF-ratio ~0.03."""
import re, glob, numpy as np
D = "/scratch/rnga/vvpshenov/DCE_NIK"; A = "/scratch/rnga/vvpshenov/presentation/assets"
ALPHAS = [0.0, 0.5, 1.0, 2.0]

held = {}
for lg in glob.glob(f"{D}/logs/slurm-nik-radial-*.out"):
    txt = open(lg).read()
    ma = re.search(r"radial_alpha=([\d.]+)", txt); mh = re.search(r"restored best \(heldout ([\d.eE+-]+)\)", txt)
    if ma and mh: held[float(ma.group(1))] = float(mh.group(1))

def hf_ratio(img):                                  # fraction of spatial power above half-Nyquist
    img = img / (img.mean() + 1e-12)
    F = np.abs(np.fft.fftshift(np.fft.fft2(img))) ** 2
    ny, nx = img.shape; y, x = np.indices((ny, nx)); r = np.hypot(y - ny // 2, x - nx // 2).astype(int)
    ps = np.bincount(r.ravel(), F.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return ps[len(ps) // 2:].sum() / ps.sum()
cs = hf_ratio(np.abs(np.load(f"{A}/arm1_cs100_sl13.npy")).mean(-1))

refs = None
import sys; sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import nik_adapter as NA
sh = NA.load_shared("/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"); sl = NA.load_slice("...".replace("...", "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"), 13)
krad = np.asarray(sl["kdata_radial"]); vt = np.asarray(sh["view_time"]).ravel()
c0 = krad.shape[0] // 2; dc = np.sqrt((np.abs(krad[c0, :, :]) ** 2).sum(-1))
o = np.argsort(vt); ts = vt[o]; dcs = np.clip(dc[o], np.percentile(dc, 1), np.percentile(dc, 99))
nav = np.convolve(np.pad(dcs, (30, 30), mode="reflect"), np.ones(61) / 61, mode="valid")[:len(dcs)]; nav /= np.median(nav[int(.35 * len(nav)):])
def n01(a): a = np.asarray(a, float); return (a - a.min()) / (a.max() - a.min() + 1e-12)

print(f"{'alpha':>6} {'held-out':>9} {'HF-ratio':>9} {'HF/CS':>7} {'swing%':>7}   (CS-100 HF-ratio {cs:.4g})")
for a in ALPHAS:
    f = f"{D}/results_nik_radial_a{a}/nik_slice_13.npy"
    try:
        rec = np.abs(np.load(f)); st = rec.mean(-1); nt = rec.shape[-1]
        roi = st > np.quantile(st, 0.6); c = np.array([rec[..., i][roi].mean() for i in range(nt)])
        sw = (c.max() - c.min()) / c.mean() * 100; hf = hf_ratio(st)
        print(f"{a:6.1f} {held.get(a, float('nan')):9.4f} {hf:9.4g} {hf/cs:7.2f} {sw:7.1f}")
    except FileNotFoundError:
        print(f"{a:6.1f} {held.get(a, float('nan')):9.4f}   (recon pending)")
