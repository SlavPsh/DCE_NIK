import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
B = "/scratch/rnga/vvpshenov/DCE_NIK"
z = np.load(f"{B}/step2_slice21.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0); tmf = z["tmf"]; body = z["body"]
T = mf.shape[2]; dt = np.median(np.diff(tmf))
pre = tmf < 50; base = mf[:, :, pre].mean(2); peak = mf.max(2); enh = (peak - base) / (base + 1e-6)
# --- kidneys: strong-enhancement, kidney-sized, posterior components (both) ---
hi = body & (enh > np.percentile(enh[body], 93)); hi = ndi.binary_opening(hi, iterations=1)
lab, n = ndi.label(hi); cy, cx = ndi.center_of_mass(body)
cand = []
for i in range(1, n + 1):
    m = lab == i; s = int(m.sum())
    if not (60 < s < 900): continue
    yy, xx = ndi.center_of_mass(m)
    if yy < cy - 8: continue                                  # kidneys posterior (lower rows), drop anterior vessels/liver
    cand.append((s, ndi.binary_fill_holes(ndi.binary_closing(m, iterations=2)) & body))
cand.sort(key=lambda t: -t[0]); kid = np.zeros_like(body)
for _, m in cand[:2]: kid |= m
# --- cortex/medulla: corticomedullary phase (cortex enhances earlier); split by early enhancement within each kidney ---
i_cm = int(np.argmin(abs(tmf - 80)))
e_cm = (mf[:, :, i_cm] - base) / (base + 1e-6)
cortex = np.zeros_like(body); medulla = np.zeros_like(body)
klab, kn = ndi.label(kid)
for i in range(1, kn + 1):
    km = klab == i; thr = np.median(e_cm[km]); cortex |= km & (e_cm > thr); medulla |= km & (e_cm <= thr)
print("kidneys found: %d  cortex %d medulla %d whole %d" % (kn, cortex.sum(), medulla.sum(), kid.sum()))
# --- jitter source: kidney vs STATIC muscle region; respiratory spectrum ---
static = body & (enh < np.percentile(enh[body], 15)) & (~kid)   # low-enhancement stable tissue
static = ndi.binary_erosion(static, iterations=1)
def cur(m): return np.median(mf[m], 0)
def jit(m):
    c = cur(m); late = tmf > 120; cc = c[late]; return float(np.std(np.diff(cc)) / (np.median(cc) + 1e-9))
print("late-plateau frame-jitter:  kidney %.4f  static-muscle %.4f (n=%d)  body-mean %.4f" % (
    jit(kid), jit(static), int(static.sum()), jit(body)))
# respiratory spectrum of detrended kidney plateau
late = tmf > 120; ck = cur(kid)[late]; ck = ck - ndi.uniform_filter1d(ck, 9)
ps = np.abs(np.fft.rfft(ck)) ** 2; fr = np.fft.rfftfreq(len(ck), d=dt)
pk = 1 + np.argmax(ps[1:]); print("kidney plateau dominant freq %.3f Hz (period %.1fs)  [respiration ~0.2-0.35Hz / 3-5s]" % (fr[pk], 1 / fr[pk] if fr[pk] > 0 else np.nan))
# correlation kidney-jitter vs static-jitter (coherent global?)
jk = cur(kid)[late]; js = cur(static)[late]
print("corr(kidney plateau ripple, static-muscle ripple): %.3f" % np.corrcoef(jk - ndi.uniform_filter1d(jk, 9), js - ndi.uniform_filter1d(js, 9))[0, 1])
# overlay figure
mfmean = mf.mean(2); vmax = np.percentile(mfmean[body], 99)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, im, ttl in [(ax[0], mfmean, "model-free mean"), (ax[1], mf[:, :, i_cm], "cortical ~80s")]:
    a.imshow(im, cmap="gray", vmax=vmax); a.contour(cortex, [.5], colors="lime", linewidths=1)
    a.contour(medulla, [.5], colors="orange", linewidths=1); a.contour(static, [.5], colors="cyan", linewidths=.6); a.set_title(ttl); a.axis("off")
ax[2].plot(tmf, cur(cortex), label="cortex"); ax[2].plot(tmf, cur(medulla), label="medulla")
ax[2].plot(tmf, cur(kid), "k", lw=2, label="whole kidney"); ax[2].plot(tmf, cur(static), "c", label="static muscle")
ax[2].legend(); ax[2].set_title("model-free curves"); fig.suptitle("real slice 21: NEW both-kidney segmentation (green/orange), cyan=static ref")
fig.tight_layout(); o = f"{B}/results/realdata_nik_vs_cs_figures/figures/realkid_newseg.png"; fig.savefig(o, dpi=110); print("SAVED", o)
np.savez(f"{B}/realkid_slice21.npz", cortex=cortex, medulla=medulla, kidney=kid, static=static)
