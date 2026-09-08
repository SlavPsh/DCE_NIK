import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import ndimage as ndi
B = "/scratch/rnga/vvpshenov/DCE_NIK"
z = np.load(f"{B}/step2_slice21.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0); tmf = z["tmf"]; body = z["body"]
rk = np.load(f"{B}/realkid_slice21.npz"); kid = np.asarray(rk["kidney"]); cx_m = np.asarray(rk["cortex"]); md_m = np.asarray(rk["medulla"])
# light temporal smoothing for per-voxel stats (motion/noise)
mfs = ndi.uniform_filter1d(mf, 5, axis=2)
base = mfs[:, :, tmf < 50].mean(2)
first = (mfs[:, :, (tmf > 60) & (tmf < 110)].max(2) - base)          # first-pass amplitude
late = (mfs[:, :, (tmf > 200) & (tmf < 350)].mean(2) - base)         # late plateau
ratio = first / (late + 1e-9)                                        # >1 = first-pass overshoot = cortex-like
ttp = tmf[np.argmax(mfs - base[:, :, None], axis=2)]                 # per-voxel time-to-peak
# functional split within the kidney: cortex = overshoot / early TTP
kv = kid
thr = np.median(ratio[kv])
cortex_f = kv & (ratio > thr); medulla_f = kv & (ratio <= thr)
def med(m): return np.median(mf[m], 0)
def enh(c): b = c[tmf < 50].mean(); return c - b
print("kidney voxels %d | ratio(first/late) median %.2f range[%.2f,%.2f] | TTP median %.0fs" % (kv.sum(), thr, ratio[kv].min(), ratio[kv].max(), np.median(ttp[kv])))
print("morphological rim/core curve difference (RMS of cortex-medulla enh): %.4e" % np.sqrt(np.mean((enh(med(cx_m)) - enh(med(md_m))) ** 2)))
print("functional split curve difference (RMS):                            %.4e" % np.sqrt(np.mean((enh(med(cortex_f)) - enh(med(medulla_f))) ** 2)))
print("functional cortex TTP %.0fs medulla TTP %.0fs" % (np.median(ttp[cortex_f]), np.median(ttp[medulla_f])))
# figure: overlay both splits + curves
mfm = mf.mean(2); vmax = np.percentile(mfm[body], 99.5)
ys, xs = np.where(ndi.binary_dilation(kid, iterations=6)); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
fig, ax = plt.subplots(2, 2, figsize=(13, 10))
for a, (cm, mm, ttl) in zip(ax[0], [(cx_m, md_m, "morphological rim/core"), (cortex_f, medulla_f, "functional (first-pass overshoot)")]):
    a.imshow(mfm[y0:y1, x0:x1], cmap="gray", vmax=vmax); a.axis("off"); a.set_title(ttl)
    a.contour(cm[y0:y1, x0:x1], [.5], colors="lime", lw=1.4) if False else a.contour(cm[y0:y1, x0:x1], [.5], colors="lime", linewidths=1.4)
    a.contour(mm[y0:y1, x0:x1], [.5], colors="orange", linewidths=1.4)
ax[1, 0].plot(tmf, enh(med(cx_m)), "lime", label="cortex(rim)"); ax[1, 0].plot(tmf, enh(med(md_m)), "orange", label="medulla(core)"); ax[1, 0].legend(); ax[1, 0].set_title("morphological curves")
ax[1, 1].plot(tmf, enh(med(cortex_f)), "lime", label="cortex(func)"); ax[1, 1].plot(tmf, enh(med(medulla_f)), "orange", label="medulla(func)"); ax[1, 1].legend(); ax[1, 1].set_title("functional curves")
fig.suptitle("real slice 21: morphological vs functional cortex/medulla split (model-free)"); fig.tight_layout()
o = f"{B}/results/realdata_nik_vs_cs_figures/figures/kidney_split_test.png"; fig.savefig(o, dpi=115); print("SAVED", o)
np.savez(f"{B}/realkid_func_slice21.npz", cortex=cortex_f, medulla=medulla_f, kidney=kid)
