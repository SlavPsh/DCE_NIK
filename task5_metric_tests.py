"""TASK 5 Parts 8-9: deterministic unit tests for the MINIMAL metric set (NRMSE + one structural
metric SSIM for images; NRMSE + first-pass peak-time/FWHM for curves). Small synthetic arrays,
truth-derived data range, shared scale. Confirms metric responses are sensible / monotone.
NOTE: skimage absent in torch29 -> windowed SSIM implemented with scipy.ndimage (Wang et al. 2004)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, csv
from scipy.ndimage import uniform_filter, gaussian_filter, shift as ndshift
OUT = "/scratch/rnga/vvpshenov/DCE_NIK/results/task5_evaluation_code_audit"
rng = np.random.default_rng(0)

def nrmse(a, b, mask=None, rng_val=None):          # truth-range normalized, shared scale
    m = np.ones_like(b, bool) if mask is None else mask
    r = (b[m].max() - b[m].min()) if rng_val is None else rng_val
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)) / (r + 1e-12))

def ssim(a, b, rng_val, win=7):                    # Wang 2004, uniform window
    a = a.astype(float); b = b.astype(float); C1 = (0.01 * rng_val) ** 2; C2 = (0.03 * rng_val) ** 2
    mu_a = uniform_filter(a, win); mu_b = uniform_filter(b, win)
    va = uniform_filter(a * a, win) - mu_a ** 2; vb = uniform_filter(b * b, win) - mu_b ** 2
    vab = uniform_filter(a * b, win) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * vab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2))
    return float(s.mean())

# ---------- IMAGE truth: structured phantom ----------
N = 96; yy, xx = np.mgrid[0:N, 0:N]; truth = np.zeros((N, N))
truth[(yy - 40) ** 2 + (xx - 40) ** 2 < 22 ** 2] = 1.0        # organ
truth[(yy - 45) ** 2 + (xx - 55) ** 2 < 6 ** 2] = 1.6         # bright small vessel
truth[(yy - 30) ** 2 + (xx - 35) ** 2 < 10 ** 2] = 0.6        # medulla
body = truth > 0; R = truth.max() - truth.min()
def streak(im):
    s = im.copy()
    for a in np.linspace(0, np.pi, 12): s += 0.15 * np.sin((xx * np.cos(a) + yy * np.sin(a)) * 1.5)
    return s
cases = {
    "identical": truth.copy(),
    "scale_x1.1": truth * 1.1,
    "offset_+0.1": truth + 0.1,
    "gaussian_noise_0.1": truth + rng.normal(0, 0.1, truth.shape),
    "shift_1px": ndshift(truth, (1, 0), order=1),
    "blur_sigma2": gaussian_filter(truth, 2.0),
    "streak": streak(truth),
    "zero_bg_noise(bg only)": truth + (~body) * rng.normal(0, 0.5, truth.shape),
}
rows = []
for name, im in cases.items():
    rows.append(dict(test=name,
        NRMSE_global=round(nrmse(im, truth, rng_val=R), 4),
        NRMSE_bodymask=round(nrmse(im, truth, mask=body, rng_val=R), 4),
        SSIM=round(ssim(im, truth, R), 4)))
with open(f"{OUT}/image_metric_unit_tests.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
print("IMAGE metric tests:")
for r in rows: print(" ", r)
# sanity assertions
assert rows[0]["NRMSE_global"] == 0.0 and abs(rows[0]["SSIM"] - 1.0) < 1e-6, "identical must give NRMSE 0 / SSIM 1"
zbg = [r for r in rows if "zero_bg" in r["test"]][0]
print(f"  [zero-bg] global NRMSE {zbg['NRMSE_global']} vs body-mask NRMSE {zbg['NRMSE_bodymask']} -> mask removes background domination:", zbg["NRMSE_bodymask"] < zbg["NRMSE_global"])

# ---------- CURVE truth: first-pass bolus + tissue uptake ----------
t = np.linspace(0, 180, 181)                                  # seconds (physical time)
def gammavar(t, t0, a, b, A):
    x = np.clip(t - t0, 0, None); return A * (x ** a) * np.exp(-x / b)
aif = gammavar(t, 10, 3.0, 4.0, 1.0); aif = aif / aif.max()   # sharp first pass
def fwhm(c):
    pk = c.max(); half = pk / 2; idx = np.where(c >= half)[0]
    return float(t[idx[-1]] - t[idx[0]]) if idx.size > 1 else 0.0
def peaktime(c): return float(t[np.argmax(c)])
def cnrmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)) / (b.max() - b.min() + 1e-12))
shifted = gammavar(t, 18, 3.0, 4.0, 1.0); shifted = shifted / shifted.max()   # same shape/amplitude, peak +8s
ccases = {
    "exact": aif.copy(),
    "time_shift_+8s": shifted,
    "amp_scale_x1.2": aif * 1.2,
    "temporal_smooth": gaussian_filter(aif, 4),
    "oscillatory_noise": aif + 0.05 * np.sin(2 * np.pi * 0.25 * t),
    "truncated_first_pass": np.where(t < 30, aif, aif[np.argmin(np.abs(t - 30))]),
}
crows = []
for name, c in ccases.items():
    crows.append(dict(test=name, curve_NRMSE=round(cnrmse(c, aif), 4), peak_time_s=round(peaktime(c), 1),
                      true_peak_time_s=round(peaktime(aif), 1), FWHM_s=round(fwhm(c), 1), true_FWHM_s=round(fwhm(aif), 1)))
with open(f"{OUT}/curve_metric_unit_tests.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(crows[0])); w.writeheader(); [w.writerow(r) for r in crows]
print("CURVE metric tests:")
for r in crows: print(" ", r)
assert crows[0]["curve_NRMSE"] == 0.0, "exact curve must give NRMSE 0"
print("METRIC_TESTS_DONE")
