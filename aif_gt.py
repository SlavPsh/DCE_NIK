"""ground-truth signal-domain AIF for F0, reproduced from the XCAT-ERIC generative model (no recon, no truth image).
plasma AIF = Cosine4AIF (mb=22.8, ae=1.36, me=0.171, t0=12s), artery tissue = ExtKety(ke=0,ve=0,vp=0.6,dt=7s)
-> C_artery = 0.6*cp(t-19s); signal = SPGR(R1=1000/1664 + 3.5*C, TR=4.66, FA=18) matching XCAT_to_MR_DCE.m."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, xph_pipeline as P, xph_common as X

def special_cosine_exp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (x * (1 - np.cos(y)) - y * np.sin(y) + y ** 2 * (1 - np.exp(-x)) / x) / (x ** 2 + y ** 2)
    return np.where(np.isfinite(f), f, 0.0)
def cosine_bolus(t, m):
    z = m * t; y = np.zeros_like(t, float); I = (z > 0) & (z < 2 * np.pi); y[I] = 1 - np.cos(z[I]); return y
def conv_bolus_exp(t, m, k):
    tB = 2 * np.pi / m; y = np.zeros_like(t, float); I1 = (t > 0) & (t < tB); I2 = t >= tB
    y[I1] = t[I1] * special_cosine_exp(k * t[I1], m * t[I1])
    y[I2] = tB * special_cosine_exp(k * tB, m * tB) * np.exp(-k * (t[I2] - tB)); return y
def cp_curve(t_min, ab, mb, ae, me, t0):                     # Cosine4AIF plasma curve, t0 in min
    tt = t_min - t0
    return ab * cosine_bolus(tt, mb) + ab * ae * conv_bolus_exp(tt, mb, me)
def spgr(T1_ms, TR=4.66, FA=18.0):
    E1 = np.exp(-TR / T1_ms); a = np.deg2rad(FA)
    return np.sin(a) * (1 - E1) / (1 - np.cos(a) * E1)
def props(t, c):
    c = c - np.median(c[t < 15]); pk = c.max(); pt = t[c.argmax()]; ab = np.where(c >= pk / 2)[0]
    return pt, (float(t[ab[-1]] - t[ab[0]]) if ab.size > 1 else 0.0), pk

if __name__ == "__main__":
    d = P.data(); times = d["times"].astype(float); ZI = P.ZI
    Hct = 0.4; ab = 2.84 / (1 - Hct); mb = 22.8; ae = 1.36; me = 0.171
    t0_art = (12.0 + 7.0) / 60.0                              # injection 12s + artery transit dt 7s
    vp = 1.0 - Hct; relax = 3.5; T1_art = 1664.0              # artery, 3T
    C_art = vp * cp_curve(times / 60.0, ab, mb, ae, me, t0_art)   # mM
    R1 = 1000.0 / T1_art + relax * C_art; S = spgr(1000.0 / R1)   # SPGR signal
    aif = S - S[times < 15].mean(); aif = np.clip(aif, 0, None); aifn = aif / (aif.max() + 1e-9)   # signal enhancement, normalized
    integ = np.concatenate([[0], np.cumsum(0.5 * (aifn[1:] + aifn[:-1]) * np.diff(times))]); integ /= (integ.max() + 1e-9)
    Tr = X.truth_at(ZI, times); R = X.rois(ZI, d["labels"]); ta = np.median(Tr[R["aorta"]], 0)
    for nm, c in [("GT signal-domain AIF (analytic)", aifn * aif.max()), ("truth aorta ROI (signal)", ta)]:
        pt, fw, _ = props(times, c); print("%-34s peak %.1fs  FWHM %.1fs  enh %.1fx" % (nm, pt, fw, (c.max() - np.median(c[times < 15])) / (np.median(c[times < 15]) + 1e-9) + 1))
    tail = slice(aifn.argmax() + 3, len(aifn)); nturn = int((np.diff(np.sign(np.diff(aifn[tail]))) != 0).sum())
    # correlation of shape vs truth aorta (normalized enhancement)
    tan = (ta - np.median(ta[times < 15])); tan = tan / (tan.max() + 1e-9)
    print("GT-AIF tail turning-points: %d | shape corr vs truth aorta: %.4f" % (nturn, np.corrcoef(aifn, tan)[0, 1]))
    np.savez("aif_xph.npz", aif_frame=aifn.astype(np.float32), tC=times.astype(np.float32), integ=integ.astype(np.float32), source="xcat-gt-signal-aif", zi=ZI)
    print("saved aif_xph.npz (GT signal-domain AIF)")
