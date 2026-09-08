import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_pipeline as P, xph_common as X
A = f"{X.OUT}/arrays"
aif = np.load("aif_xph.npz"); af = aif["aif_frame"].astype(float); tC = aif["tC"].astype(float)
d = P.data(); tq = d["times"]; R = X.rois(P.ZI, d["labels"]); Tr = X.truth_at(P.ZI, tq)
f0 = np.load(f"{A}/nik_eval_w768_ks2.5_s0.npz")["rec_best"]
ta = np.median(Tr[R["aorta"]], 0); fa = np.median(f0[R["aorta"]], 0)          # truth vs F0 aorta (median)
# detrend f0 aorta (subtract smooth) to expose ripple
from scipy.ndimage import uniform_filter1d
fa_s = uniform_filter1d(fa, 15); af_s = uniform_filter1d(af, 15)
fa_rip = fa - fa_s; af_rip = af - af_s
print("AIF: n=%d  peak %.1fs  min-gap(non-monotone tail?) " % (len(af), tC[af.argmax()]))
# count local extrema in the AIF tail (post-peak) -> ripple
tail = slice(af.argmax()+3, len(af))
dsign = np.diff(np.sign(np.diff(af[tail])))
n_turn = int((dsign != 0).sum())
print("AIF post-peak local turning points:", n_turn, " (smooth curve -> ~0-1)")
# dominant ripple period via FFT of detrended AIF tail
seg = af_rip[tail]; seg = seg - seg.mean()
if len(seg) > 8:
    ps = np.abs(np.fft.rfft(seg))**2; fr = np.fft.rfftfreq(len(seg), d=np.median(np.diff(tC)))
    k = 1 + np.argmax(ps[1:]); per = 1/fr[k] if fr[k] > 0 else np.nan
    print("AIF ripple dominant period: %.1f s  (30-bin AIF -> ~%.1f s/bin)" % (per, (tC[-1]-tC[0])/30))
# correlation of F0-aorta ripple with AIF ripple
m = np.isfinite(fa_rip) & np.isfinite(af_rip)
print("corr(F0-aorta ripple, AIF ripple): %.3f" % np.corrcoef(fa_rip[m], af_rip[m])[0,1])
fig, ax = plt.subplots(2, 1, figsize=(11, 7))
ax[0].plot(tC, af, lw=1.5, label="AIF (aif_frame)"); ax[0].plot(tq, ta/ta.max(), "k--", lw=1, label="truth aorta (norm)")
ax[0].plot(tq, fa/fa.max(), lw=1, label="F0 aorta (norm)"); ax[0].legend(fontsize=8); ax[0].set_title("aif vs curves")
ax[1].plot(tC, af_rip, label="AIF ripple"); ax[1].plot(tq, fa_rip/ (fa.max()+1e-9)*(af.max()+1e-9), label="F0-aorta ripple (scaled)")
ax[1].plot(tq, (ta-uniform_filter1d(ta,15)), "k--", lw=1, label="truth aorta ripple"); ax[1].legend(fontsize=8); ax[1].set_title("ripple overlay")
fig.tight_layout(); fig.savefig(f"{X.OUT}/figures/aif_ripple_check.png", dpi=110); print("SAVED aif_ripple_check.png")
