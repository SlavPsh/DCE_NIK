"""P3 STEP 3.1: estimate the AIF from the slice-21 aorta ROI (streak-free, ROI-averaged
model-free) and run the automated plausibility gate. if it FAILS, P3 is skipped (a bad AIF
corrupts every downstream amplitude). writes aif_slice21.npz + PASS/FAIL verdict."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, scipy.ndimage as ndi, json
from scipy.signal import savgol_filter, find_peaks
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
from figpath import fig as fpath
REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; D = "/net/beegfs/users/P101440/DCE_NIK"; TA = 375.0
Z = int(sys.argv[1]) if len(sys.argv) > 1 else 21

d = np.load(f"{D}/step2_slice{Z}.npz"); mf = d["mf"]; tmf = d["tmf"]
cs = np.abs(np.load(f"{REF}/slice_{Z:02d}.npz")["cs_img"]).astype(np.float32); tC = np.linspace(0, TA, cs.shape[-1])
base = cs[..., tC < 50].mean(-1); enh = cs - base[..., None]; m = cs.mean(-1); body = m > np.quantile(m, 0.5)
late = enh[..., (tC > 130) & (tC < 200)].mean(-1); kid = body & (late > np.quantile(late[body], 0.985)); kid = ndi.binary_opening(kid, iterations=1)
l, k = ndi.label(kid); kid = (l == (1 + np.argmax(ndi.sum(np.ones_like(l), l, range(1, k + 1))))) if k else kid
early = enh[..., (tC > 40) & (tC < 80)].mean(-1); ao = body & (early > np.quantile(early[body], 0.995)) & (~kid); ao = ndi.binary_opening(ao, iterations=1)
la, ka = ndi.label(ao); ao = (la == (1 + np.argmax(ndi.sum(np.ones_like(la), la, range(1, ka + 1))))) if ka else ao

# raw (unnormalized) aorta AIF, model-free
aif = np.array([im[ao].mean() for im in mf]); b0 = aif[tmf < 45].mean(); aif = aif - b0
sm = savgol_filter(aif, 11, 3); pk = sm.max(); sm_n = sm / (pk + 1e-9); raw_n = aif / (pk + 1e-9)

# --- automated plausibility checks ---
ttp = float(tmf[np.argmax(sm)])
onset_i = int(np.argmax(sm > 0.1 * pk)); onset_t = float(tmf[onset_i])
# monotonic onset: from onset to peak, curve should be (mostly) increasing
seg = sm[onset_i:np.argmax(sm) + 1]; mono_frac = float(np.mean(np.diff(seg) >= -0.02 * pk)) if len(seg) > 2 else 0.0
# sharp first pass: rise time 10->90% of peak
r10 = tmf[np.argmax(sm > 0.1 * pk)]; r90 = tmf[np.argmax(sm > 0.9 * pk)]; rise = float(r90 - r10)
# no negative garbage at baseline
neg = float(raw_n[tmf < 45].min())
# recirculation bump: a secondary peak after the first-pass trough
after = np.argmax(sm); tail = sm_n[after:];
trough_rel = np.argmin(tail[:max(2, len(tail) // 3)]) if len(tail) > 4 else 0
recirc = False; rc_amp = 0.0
if len(tail) > trough_rel + 3:
    peaks, _ = find_peaks(tail[trough_rel:], prominence=0.03)
    recirc = len(peaks) > 0; rc_amp = float(tail[trough_rel:][peaks].max() - tail[trough_rel:].min()) if len(peaks) else 0.0

checks = {
    "sharp_first_pass_rise<40s": rise < 40,
    "physiological_TTP_45-110s": 45 <= ttp <= 110,
    "monotonic_onset>0.8": mono_frac > 0.8,
    "no_negative_baseline>-0.15": neg > -0.15,
    "recirculation_bump": bool(recirc),
}
# recirculation is desirable but not disqualifying; the hard gates are the other four
hard = [v for kk, v in checks.items() if kk != "recirculation_bump"]
verdict = "PASS" if all(hard) else "FAIL"
print("===== AIF GATE slice 21 =====")
print(f"aorta ROI {int(ao.sum())} vox | TTP {ttp:.0f}s onset {onset_t:.0f}s rise10-90 {rise:.0f}s mono {mono_frac:.2f} baseNeg {neg:.2f} recirc {recirc}(amp {rc_amp:.2f})")
for kk, v in checks.items(): print(f"  [{'OK' if v else 'XX'}] {kk}")
print(f"VERDICT: {verdict}  ({'proceed to P3' if verdict=='PASS' else 'SKIP P3, do not feed bad AIF downstream'})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
a = cs[..., int(np.argmin(np.abs(tC - 60)))]; ov = np.zeros((*a.shape, 4)); ov[ao] = [1, 1, 0, .7]
ax[0].imshow(np.rot90(a), cmap="gray", vmax=np.percentile(a, 99.5)); ax[0].imshow(np.rot90(ov)); ax[0].axis("off"); ax[0].set_title(f"slice {Z} aorta ROI ({int(ao.sum())} vox)")
ax[1].plot(tmf, raw_n, "0.6", lw=1, label="model-free"); ax[1].plot(tmf, sm_n, "r", lw=2, label="smoothed AIF")
ax[1].axvline(ttp, color="b", ls="--", lw=1, label=f"TTP {ttp:.0f}s"); ax[1].set_xlim(0, 260); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
ax[1].set_title(f"AIF gate: {verdict}  (rise {rise:.0f}s, mono {mono_frac:.2f})")
fig.suptitle(f"P3 AIF estimate + plausibility gate, slice {Z}", fontweight="bold"); fig.tight_layout()
p = fpath("aif_gate_slice21.png"); fig.savefig(p, dpi=135); print("wrote", p.split("/")[-1])

# save AIF on the CS frame grid (interp) for the Patlak basis, only meaningful if PASS
aif_frame = np.interp(tC, tmf, sm); aif_frame = np.maximum(aif_frame, 0)
np.savez(f"{D}/aif_slice{Z}.npz", aif_tmf=aif, tmf=tmf, aif_frame=aif_frame, tC=tC, ao=ao,
         verdict=verdict, ttp=ttp, checks=json.dumps(checks))
