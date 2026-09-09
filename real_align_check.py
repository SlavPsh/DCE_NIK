"""Real-data blast-radius check: is the in-vivo spoke-frontier (NIK HaarPSI below CS at every fraction)
partly a NIK-vs-CS RELATIVE alignment artifact, the analog of the phantom 1px bug? NIK (cufinufft render)
and CS (grasp_pro render) are different code paths on a 192 (even) grid. Measure the relative shift +
orientation of NIK vs the CS-f100 anchor, then RE-SCORE HaarPSI after aligning NIK. If the NIK-vs-CS gap
shrinks, the frontier was partly misalignment."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, piq
from scipy.ndimage import fourier_shift
from scipy.interpolate import interp1d
import recon_asserts as RA
D = "/net/beegfs/users/P101440/DCE_NIK"; CSD = "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs"
ref = np.abs(np.load(f"{CSD}/cs_slice13_f100.npy")).astype(np.float32); NT = ref.shape[-1]        # [192,192,122]
rm = ref.mean(-1); body = rm > np.quantile(rm, 0.55); cm = rm
def resample_t(v, nt):
    if v.shape[-1] == nt: return v
    return interp1d(np.linspace(0, 1, v.shape[-1]), v, axis=-1, kind="linear")(np.linspace(0, 1, nt)).astype(np.float32)
def to_batch(v):
    v = v * body[:, :, None]; v = v / (np.percentile(v, 99.5) + 1e-9); v = np.clip(v, 0, 1)
    return torch.from_numpy(np.transpose(v, (2, 0, 1))[:, None]).float()
rb = to_batch(ref)
def haar(v):
    with torch.no_grad(): return piq.haarpsi(to_batch(resample_t(v, NT)), rb, reduction="none", data_range=1.0).cpu().numpy()
def fshift_vol(v, sy, sx): return np.stack([np.real(np.fft.ifft2(fourier_shift(np.fft.fft2(v[:, :, t]), (sy, sx)))) for t in range(v.shape[2])], -1).astype(np.float32)
def orient(im, nm): return {"id": im, "rot180": im[::-1, ::-1], "fliplr": im[:, ::-1], "flipud": im[::-1], "T": im.T, "rot90": np.rot90(im), "rot270": np.rot90(im, 3)}[nm]
def ncc(a, c): a = a[body]-a[body].mean(); c = c[body]-c[body].mean(); return float((a*c).sum()/(np.linalg.norm(a)*np.linalg.norm(c)+1e-12))

# --- measure NIK-f100 vs CS-f100 relative alignment ---
nik100 = np.abs(np.load(f"{D}/results_spoke_nik_f100/nik_slice_13.npy")).astype(np.float32); nm = nik100.mean(-1)
best = max(["id", "rot180", "fliplr", "flipud", "T", "rot90", "rot270"], key=lambda k: ncc(orient(nm, k), cm) if orient(nm, k).shape == cm.shape else -9)
sy, sx, cc = RA.measure_shift(nm, cm, body, rng=3.0)
print(f"NIK-f100 vs CS-f100: best D4 = {best} (ncc {ncc(orient(nm,best),cm):+.3f}); relative sub-pixel shift (sy,sx) = ({sy:+.3f},{sx:+.3f}) ncc {cc:.3f}\n")

# --- re-score the frontier: raw vs aligned NIK ---
print(f"{'fraction':10s} {'NIK raw':>9s} {'NIK aligned':>12s} {'CS':>9s}  (HaarPSI vs CS-f100, mean)")
for lab in ["f100", "f70", "f50", "f35", "f25"]:
    try:
        nik = np.abs(np.load(f"{D}/results_spoke_nik_{lab}/nik_slice_13.npy")).astype(np.float32)
        cs = np.abs(np.load(f"{CSD}/cs_slice13_{lab}.npy")).astype(np.float32)
    except FileNotFoundError: continue
    hr = haar(nik).mean(); ha = haar(fshift_vol(nik, sy, sx)).mean(); hc = haar(cs).mean()
    print(f"{lab:10s} {hr:9.4f} {ha:12.4f} {hc:9.4f}", flush=True)
print("\nif NIK aligned >> NIK raw and closes on CS -> frontier was partly a NIK-vs-CS misalignment artifact.")
print("REAL_ALIGN_DONE")
