"""parametric aif for the tofts basis (FINDINGS 9, lever 1): the measured aorta curve is the true aif convolved with the 31-spoke window
(a box of W = 31 * TA / ntviews seconds, 6.8 s on p3) plus noise. fit C(t) = A1 gv(t - t0; a1, b1) + A2 gv(t - t0 - d; a2, b2) + A3 (1 - exp(-(t - t0) / tr))
exp(-(t - t0) / tw) (first pass, recirculation, slow washout; gv = peak-normalized gamma variate) such that box_W * C matches the RAW roi curve
in least squares; the deconvolved C is the basis input (noise-free, first pass unblurred). consistency check: the predicted 11-spoke / 31-spoke
peak ratio of C must agree with the measured factor of mf_peak_check. writes aif_param{_ds}_slice<Z>.npz (same keys as aif_gate) + figure.
usage: DCE_DS=p3 python aif_param_fit.py --slice 21"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, argparse, numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp

def gv(t, a, b):                                                                   # gamma variate, peak 1 at t = a b, zero for t <= 0
    t = np.maximum(t, 0.0); return np.where(t > 0, (t / (a * b + 1e-12)) ** a * np.exp(a - t / (b + 1e-12)), 0.0)

def model(p, t):
    A1, t0, a1, b1, A2, d, a2, b2, A3, tr, tw = p; u = t - t0
    return A1 * gv(u, a1, b1) + A2 * gv(u - d, a2, b2) + A3 * np.where(u > 0, (1 - np.exp(-np.maximum(u, 0) / tr)) * np.exp(-np.maximum(u, 0) / tw), 0.0)

def box(c, tf, W):                                                                 # moving average of width W s on the fine grid
    n = max(1, int(round(W / (tf[1] - tf[0])))); k = np.ones(n) / n; return np.convolve(np.pad(c, (n // 2, n - 1 - n // 2), mode="edge"), k, mode="valid")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); ap.add_argument("--spokes", type=int, default=31); a = ap.parse_args(); Z = a.slice; TA = dsp.TA
    z = np.load(dsp.AIF(Z), allow_pickle=True); raw = np.asarray(z["aif_tmf"], float); tmf = np.asarray(z["tmf"], float); tC = np.asarray(z["tC"], float)
    W = a.spokes * TA / dsp.NTV; tf = np.arange(0.0, TA + 0.05, 0.1)
    sm = savgol_filter(raw, 11, 3); pk = float(sm.max()); ttp = float(tmf[np.argmax(sm)]); onset = float(tmf[np.argmax(sm > 0.1 * pk)])
    def resid(p): return np.interp(tmf, tf, box(model(p, tf), tf, W)) - raw
    best = None
    for t0 in (onset - 4.0, onset - 2.0, onset):
        for a1 in (2.0, 4.0, 8.0):
            b1 = max(0.5, (ttp - t0) / a1)
            p0 = [pk * 1.2, t0, a1, b1, 0.3 * pk, 18.0, 3.0, 6.0, 0.35 * pk, 8.0, 300.0]
            lo = [0, onset - 15, 0.5, 0.2, 0, 5, 0.5, 0.5, 0, 1, 30]; hi = [5 * pk, ttp, 30, 60, 2 * pk, 60, 30, 60, 2 * pk, 120, 5000]
            try:
                r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=4000)
                if best is None or r.cost < best.cost: best = r
            except Exception as e: print("fit failed", p0, e)
    p = best.x; C = np.maximum(model(p, tf), 0.0); Cb = box(C, tf, W)
    pred_ratio = float(box(C, tf, 11 * TA / dsp.NTV).max() / (Cb.max() + 1e-12)); gain = float(C.max() / (Cb.max() + 1e-12))
    meas = None; pc = f"{B}/results/tofts_vs_patlak/mf_peak_check{dsp.SFX}_sl{Z}.json"
    if os.path.exists(pc):
        o = json.load(open(pc)); r11 = o["11"]["aorta"]["peak"] / o["31"]["aorta"]["peak"]; meas = float(r11 * 31.0 / 11.0 if r11 < 0.6 else r11)   # legacy json: peaks scale with the window
    rms_fit = float(np.sqrt(np.mean(resid(p) ** 2)) / (pk + 1e-12)); rms_sg = float(np.sqrt(np.mean((sm - raw) ** 2)) / (pk + 1e-12))
    aif_frame = np.maximum(np.interp(tC, tf, C), 0.0)
    out = f"{B}/aif_param{dsp.SFX}_slice{Z}.npz"
    np.savez(out, aif_tmf=raw, tmf=tmf, aif_frame=aif_frame, tC=tC, ao=z["ao"], aif_frame_conv=np.interp(tC, tf, Cb), params=p, window_s=W, peak_gain=gain, pred_ratio_11=pred_ratio,
             meas_ratio_11=(meas if meas is not None else np.nan), verdict=str(z["verdict"]) if "verdict" in z.files else "n/a", ttp=float(tf[np.argmax(C)]), source=dsp.AIF(Z))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for k, (xl, ttl) in enumerate(((0, TA), (max(0, onset - 15), onset + 60))):
        ax[k].plot(tmf, raw / pk, "0.6", lw=0.8, label="measured (31-spoke roi curve, raw)"); ax[k].plot(tmf, sm / pk, "b", lw=1.2, label="savgol 11 (current basis input)")
        ax[k].plot(tf, Cb / pk, "g", lw=1.2, label=f"fit convolved with the {W:.1f} s window (rms {rms_fit:.3f})"); ax[k].plot(tf, C / pk, "r", lw=1.4, label=f"fit deconvolved = new basis input (peak x{gain:.2f})")
        ax[k].set_xlim(*xl); ax[k].set_xlabel("t [s]"); ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=7); ax[1].set_title(f"predicted 11-spoke / 31-spoke peak ratio {pred_ratio:.2f} vs measured {meas if meas is None else round(meas, 2)}", fontsize=9)
    fig.suptitle(f"{dsp.DS} slice {Z}: parametric aif (gamma variate first pass + recirculation + washout), box-deconvolved; savgol residual rms {rms_sg:.3f}", fontsize=10); fig.tight_layout()
    fp = f"{B}/results/realdata_nik_vs_cs_figures/figures/aif_param{dsp.SFX}_slice{Z}.png"; fig.savefig(fp, dpi=130, facecolor="white")
    print(f"params {np.round(p, 3).tolist()}"); print(f"fit rms {rms_fit:.4f} (savgol {rms_sg:.4f}) | peak gain {gain:.3f} | predicted 11/31 ratio {pred_ratio:.3f} vs measured {meas} | ttp {tf[np.argmax(C)]:.1f} s | wrote {out} {fp}"); print("AIF_PARAM_DONE")

if __name__ == "__main__": main()
