"""SPIRiT evaluation. mechanism prediction: NIK's denoising = suppression of coil-INCONSISTENT
broadband noise (Task S). so an explicit coil-consistency prior (SPIRiT) should be REDUNDANT on
NIK -- inert, no fidelity/spatial change -- because NIK already enforces coil consistency
implicitly. compare full-rank sl21 baseline (weight 0) vs spirit {1e-2, 1e-1}.
out: task_spirit.json + printed table."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, os, sys
from scipy.signal import savgol_filter
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); sys.path.insert(0, ".")
import consolidated as C
D = "/scratch/rnga/vvpshenov/DCE_NIK"; TA = 375.0; Z = 21; ROIS = ["aorta", "cortex", "medulla", "liver"]
RUNS = [("w0_base", f"{D}/results_batch/full_sl21f25/nik_slice_21_cplx.npy"),
        ("w0.01", f"{D}/results_batch/spirit_w0.01_sl21/nik_slice_21_cplx.npy"),
        ("w0.1", f"{D}/results_batch/spirit_w0.1_sl21/nik_slice_21_cplx.npy")]
ctx = C.slice_ctx(Z); rois = ctx["rois"]; BODY = ctx["BODY"]; mf = np.load(f"{D}/step2_slice{Z}.npz")["mf"]; tmf = ctx["tmf"]
names = [r for r in ROIS if rois.get(r) is not None and rois[r].sum() > 0]
real = {r: np.array([im[rois[r]].mean() for im in mf]) for r in names}
def osc(c): sm = savgol_filter(c, 11, 3); return float(np.std(c - sm) / (abs(sm).max() + 1e-9))
def relres(p, t): return float(np.sqrt(np.mean((p - t) ** 2)) / (t.max() - t.min() + 1e-9))

out = {}
for tag, path in RUNS:
    if not os.path.exists(path): out[tag] = dict(missing=True); print("MISSING", tag); continue
    cplx = np.load(path); mag = np.abs(cplx).astype(np.float32); tN = np.linspace(0, TA, mag.shape[-1])
    X = cplx[BODY]; Xc = X - X.mean(1, keepdims=True); sv = np.linalg.svd(Xc, compute_uv=False)
    ev = np.cumsum(sv**2)/np.sum(sv**2); rrank = int(np.searchsorted(ev, 0.99)+1)
    sp = C.spatial_eval(mag, Z)
    cur = {r: np.interp(tmf, tN, np.array([mag[..., i][rois[r]].mean() for i in range(mag.shape[-1])])) for r in names}
    Ns = np.concatenate([cur[r] for r in names]); Rs = np.concatenate([real[r] for r in names])
    a, b = np.linalg.lstsq(np.stack([Ns, np.ones_like(Ns)], 1), Rs, rcond=None)[0]
    out[tag] = dict(rrank=rrank, rulerA=sp["rulerA_haarpsi"], rulerB=sp["rulerB_haarpsi"], bgE=sp["bgE"],
                    osc={r: osc(cur[r]) for r in names}, fid={r: relres(a*cur[r]+b, real[r]) for r in names})

print("=== SPIRiT on full-rank NIK sl21: is the coil-consistency prior redundant (inert)? ===")
print(f"{'run':>9}{'rrank':>6}{'rulerA':>8}{'rulerB':>8}{'bgE':>7}{'osc_aorta':>10}{'osc_cortex':>11}{'fid_cortex':>11}")
for tag, _ in RUNS:
    r = out.get(tag, {})
    if r.get("missing"): print(f"{tag:>9}  MISSING"); continue
    print(f"{tag:>9}{r['rrank']:>6}{r['rulerA']:>8.3f}{r['rulerB']:>8.3f}{r['bgE']:>7.3f}{r['osc']['aorta']:>10.3f}{r['osc']['cortex']:>11.3f}{r['fid']['cortex']:>11.3f}")
b = out.get("w0_base", {})
if b and not b.get("missing"):
    print("\ndeltas vs baseline (should be ~0 if SPIRiT is redundant on NIK):")
    for tag, _ in RUNS[1:]:
        r = out.get(tag, {})
        if r.get("missing"): continue
        print(f"  {tag}: dRulerA {r['rulerA']-b['rulerA']:+.3f}  dBgE {r['bgE']-b['bgE']:+.3f}  dOsc_aorta {r['osc']['aorta']-b['osc']['aorta']:+.3f}  dFid_cortex {r['fid']['cortex']-b['fid']['cortex']:+.3f}")
json.dump(out, open(f"{D}/task_spirit.json", "w"), indent=1, default=float)
print("\nREAD: near-zero deltas -> SPIRiT redundant on NIK (NIK already coil-consistent) -> supports the denoising-as-coil-inconsistent-noise-removal picture.")
