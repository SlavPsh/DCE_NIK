import warnings; warnings.filterwarnings("ignore")
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
B = "/scratch/rnga/vvpshenov/DCE_NIK"; GP = "/scratch/rnga/vvpshenov/grasp_pro_py"
z = np.load(f"{B}/step2_slice21.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0); tmf = z["tmf"]; body = z["body"]  # [192,192,T]
gk = np.load(f"{B}/gate_slice21.npz"); cortex = np.asarray(gk["cortex"]); medulla = np.asarray(gk["medulla"])
kid = cortex | medulla
mfmean = mf.mean(2); vmax = np.percentile(mfmean[body], 99)
fcm = int(np.argmin(abs(tmf - 85)))                     # cortical phase
def curve(m): return np.median(mf[m], 0)
def jitter(m):                                          # frame-to-frame osc / plateau, over late plateau
    c = curve(m); late = tmf > 120; cc = c[late]; return float(np.std(np.diff(cc)) / (np.median(cc) + 1e-9))
# whole-kidney (dilated) vs current cortex/medulla jitter
kid_big = ndi.binary_dilation(kid, iterations=2) & body
kid_core = ndi.binary_erosion(kid, iterations=1)
print("ROI sizes: cortex %d medulla %d kidney %d  kid_big %d kid_core %d" % (cortex.sum(), medulla.sum(), kid.sum(), kid_big.sum(), kid_core.sum()))
for nm, m in [("cortex", cortex), ("medulla", medulla), ("whole-kidney", kid), ("kidney dilated", kid_big), ("kidney core(eroded)", kid_core)]:
    print("  %-20s n=%4d  late-plateau frame-jitter %.4f" % (nm, int(m.sum()), jitter(m)))
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a, im, ttl in [(ax[0], mfmean, "model-free mean"), (ax[1], mf[:, :, fcm], "cortical ~85s")]:
    a.imshow(im, cmap="gray", vmax=vmax); a.contour(cortex, levels=[.5], colors="lime", linewidths=1)
    a.contour(medulla, levels=[.5], colors="orange", linewidths=1); a.set_title(ttl); a.axis("off")
ax[2].plot(tmf, curve(cortex), label="cortex"); ax[2].plot(tmf, curve(medulla), label="medulla")
ax[2].plot(tmf, curve(kid), label="whole kidney", lw=2, color="k"); ax[2].legend(); ax[2].set_title("model-free curves (jitter)")
fig.suptitle("real slice 21: current kidney ROIs + model-free curves"); fig.tight_layout()
o = f"{B}/results/realdata_nik_vs_cs_figures/figures/realkid_diag.png"; fig.savefig(o, dpi=110); print("SAVED", o)
