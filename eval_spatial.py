"""Corrected spatial eval (FIX 1-4). the fairness problem: the 240-spoke pre-contrast
reference uses the SAME spokes as an f100 test recon, so scoring there rewards reproducing
gridding streaks, and any streak-removing prior is PENALIZED. fixes:

FIX 1  reference must have STRICTLY more data than the test.
       - f100 DROPPED from the spatial axis (reference == test, degenerate); kept only as a
         labelled non-comparative sanity point.
       - ruler A: 240-spoke pre-contrast NUFFT, scored vs f70/f50/f25 (advantage 1.4/2.0/3.4x).
       - ruler B: all-spoke (1710) temporal-mean NUFFT, the only streak-free ruler, scored on
         NON-ENHANCING tissue only (it spans the scan so enhancing regions are intensity-
         mismatched vs a pre-contrast recon). reported SEPARATELY, never merged with A.
FIX 2  background/air ROI residual energy, reference-FREE. lower is unambiguously better.
FIX 3  difference-image STRUCTURE: streaks outside support (artifact removal, good) vs
       edge-localized inside structures (signal removal, bad). classify, do not just scalar.
FIX 4  held-out spoke consistency stays SECONDARY (immune to this problem; ~0.5 noise floor).
"""
import numpy as np, torch, json, os
import scipy.ndimage as ndi
from masked_metrics import haarpsi_masked, ssim_masked

D = "/scratch/rnga/vvpshenov/DCE_NIK"; NUF = f"{D}/results_nufft"
CSD = "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs"
TA = 375.0; dev = "cuda" if torch.cuda.is_available() else "cpu"
meta = json.load(open(f"{NUF}/meta.json")); T_PRE = meta["t_pre_s"]
SPOKES = {"f100": 240, "f70": 172, "f50": 121, "f25": 70}          # pre-contrast spokes/fraction
REF_PRE_SPOKES = 240

ref_pre = np.load(f"{NUF}/nufft_pre.npy").astype(np.float32)       # ruler A, 240 spokes
ref_all = np.load(f"{NUF}/nufft_all.npy").astype(np.float32)       # ruler B, 1710-spoke mean


def _clean_body(ref):
    raw = ref > np.quantile(ref, 0.55); m0 = ndi.binary_opening(raw, iterations=2)
    l, n = ndi.label(m0)
    if n: m0 = (l == 1 + int(np.argmax(ndi.sum(np.ones_like(l), l, range(1, n + 1)))))
    return ndi.binary_closing(ndi.binary_fill_holes(m0), iterations=2)


BODY = _clean_body(ref_all)
AIR = ~ndi.binary_dilation(BODY, iterations=4)                    # FIX 2: outside support, margin
# FIX 1 ruler B support: non-enhancing tissue from a dynamic recon (low temporal variation)
_cs = np.abs(np.load(f"{CSD}/cs_slice13_f100.npy")).astype(np.float32)
_cv = _cs.std(-1) / (_cs.mean(-1) + 1e-6)
NONENH = BODY & (_cv < np.quantile(_cv[BODY], 0.5))               # static half of the body


def _ls(pred, ref, m):
    return float((pred[m] @ ref[m]) / (pred[m] @ pred[m] + 1e-12))


def _tb(img, vmax):
    return torch.from_numpy(np.clip(img / (vmax + 1e-12), 0, 1)[None, None]).float().to(dev)


def score(pred, ref, mask):
    """PSNR on mask voxels + masked HaarPSI/SSIM (similarity MAPS averaged over mask)."""
    s = _ls(pred, ref, mask); p = pred * s
    mse = float(((p[mask] - ref[mask]) ** 2).mean()); pk = float(ref[mask].max())
    psnr = 10 * np.log10(pk ** 2 / (mse + 1e-20)); vmax = float(np.percentile(ref[mask], 99.5))
    mt = torch.from_numpy(mask.astype(np.float32))[None, None].to(dev)
    with torch.no_grad():
        h = float(haarpsi_masked(_tb(p, vmax), _tb(ref, vmax), mt, data_range=1.0).cpu())
        ss = float(ssim_masked(_tb(p, vmax), _tb(ref, vmax), mt, data_range=1.0).cpu())
    return dict(psnr=psnr, haarpsi=h, ssim=ss)


def background_energy(pred, ref):
    """FIX 2: air-ROI residual energy. pred is LS-scale-matched to ref on BODY FIRST (item 2:
    otherwise bgE tracks global scale, not artifact content), then normalized by ref body RMS
    so it is comparable across methods and against the reference's own bgE (item 1)."""
    s = _ls(pred, ref, BODY)
    return float(np.sqrt(((pred * s)[AIR] ** 2).mean()) / (np.sqrt((ref[BODY] ** 2).mean()) + 1e-12))


def ref_background_energy(ref):
    """item 1: the reference's OWN air energy (no scaling, ref vs itself). the SIGNED threshold:
    recon bgE below this = removed streaks the reference has; above = added its own."""
    return float(np.sqrt((ref[AIR] ** 2).mean()) / (np.sqrt((ref[BODY] ** 2).mean()) + 1e-12))


def diff_structure(pred, ref):
    """FIX 3: WHERE the difference lives. outside-support fraction (streak/artifact removal)
    vs inside-edge fraction (signal removal). same magnitude, opposite meaning."""
    s = _ls(pred, ref, BODY); d = (pred * s - ref) ** 2
    edges = BODY & (ndi.gaussian_gradient_magnitude(ref, 1.0) >
                    np.quantile(ndi.gaussian_gradient_magnitude(ref, 1.0)[BODY], 0.8))
    tot = d.sum() + 1e-20
    return dict(outside_frac=float(d[AIR].sum() / tot),            # high => artifact removal (good)
                inside_edge_frac=float(d[edges].sum() / d[BODY].sum() + 1e-20),  # high => signal removal (bad)
                diff=np.sqrt(d))


def win_mean(v, tmax=T_PRE):
    t = np.linspace(0, TA, v.shape[-1]); return v[..., t < tmax].mean(-1)


def load_pre(method, lab):
    p = {"NUFFT": f"{NUF}/frac_pre_{lab}.npy",
         "CS": f"{CSD}/cs_slice13_{lab}.npy",
         "NIK R=16": f"{D}/results_spoke_nik_{lab}/nik_slice_13.npy",
         "NIK full": f"{D}/results_spoke_full_{lab}/nik_slice_13.npy"}[method]
    if not os.path.exists(p): return None
    v = np.abs(np.load(p)).astype(np.float32)
    return v if v.ndim == 2 else win_mean(v)


if __name__ == "__main__":
    print(f"masks: body {BODY.sum()} | air {AIR.sum()} | non-enhancing {NONENH.sum()} px")
    print(f"ruler A = 240-spoke pre (f70/f50/f25 only, advantage {240/172:.1f}/{240/121:.1f}/{240/70:.1f}x)")
    print(f"ruler B = 1710-spoke mean on non-enhancing tissue (streak-free)\n")
    # item 1: reference's OWN background energy = the signed threshold for recon bgE
    bgA = ref_background_energy(ref_pre); bgB = ref_background_energy(ref_all)
    print(f"REFERENCE bgE (threshold):  ruler A 240-sp pre = {bgA:.4f}   ruler B 1710-sp mean = {bgB:.4f}")
    print(f"  -> recon bgE BELOW {bgA:.4f} = removed streaks the reference itself has (good); above = added its own\n")
    # worked example: all methods at f25 (70 spokes), both rulers + FIX 2 (signed) + FIX 3
    print(f"{'method@f25':12} {'--- ruler A (240sp pre) ---':>30} {'ruler B (nonenh)':>18} {'bgE':>8} {'vs refA':>8} {'out%':>6} {'inEdge':>7}")
    print(f"{'':12} {'PSNR':>8}{'Haar':>8}{'SSIM':>8}   {'PSNR':>8}{'Haar':>8}   {'(FIX2)':>8}{'signed':>8}{'(FIX3)':>6}{'(loc)':>7}")
    for meth in ["NUFFT", "CS", "NIK R=16", "NIK full"]:
        pre = load_pre(meth, "f25")
        if pre is None: continue
        a = score(pre, ref_pre, BODY)                             # ruler A
        b = score(pre, ref_all, NONENH)                          # ruler B, non-enhancing only
        bg = background_energy(pre, ref_pre); sign = "clean" if bg < bgA else "ADDED"
        ds = diff_structure(pre, ref_pre)
        print(f"{meth:12} {a['psnr']:8.2f}{a['haarpsi']:8.3f}{a['ssim']:8.3f}   "
              f"{b['psnr']:8.2f}{b['haarpsi']:8.3f}   {bg:8.4f}{bg-bgA:+8.4f}{ds['outside_frac']*100:6.1f}{ds['inside_edge_frac']:7.3f}")
