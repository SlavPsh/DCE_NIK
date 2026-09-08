"""method-neutral in-vivo ruler for slice 21: held-out spokes + physical bounds.

the model-free NUFFT ruler is biased: it is an unregularized gridding recon, so a nearly
unregularized grasp v2 resembles it by construction while NIK's learned estimator does not. these
two rulers avoid that.

(A) HELD-OUT SPOKES. v%10 in {8,9} = 342 spokes neither method used. forward-project each recon to
    those locations and score. measured DATA, not another recon, so no method-family bias.
    caveat: rewards fitting noise -> consistency, not accuracy.

(B) PHYSICAL BOUNDS. reference-free, method-neutral. the aorta first-pass cannot be broader than the
    injection allows, concentration cannot be negative, and the first pass must rise monotonically.
    these can DISQUALIFY a recon outright rather than rank it.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
import numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK"); sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_v2")
import consolidated as C
from grasp_v2_py import MCNUFFT

B = "/scratch/rnga/vvpshenov/DCE_NIK"; GV = "/scratch/rnga/vvpshenov/grasp_v2/results_grasp_v2"
REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
TA, NTV = 375.0, 1710
ISIGN = -1

sh = np.load(f"{REF}/shared.npz")
Traj = np.asarray(sh["traj_grog"]); NX = Traj.shape[0]; NGRID = int(sh["nx"]); BAS = int(sh["bas"])
kdc = np.asarray(np.load(f"{REF}/slice_21.npz")["kdata_radial"]).astype(np.complex64)
b1 = np.asarray(np.load(f"{REF}/slice_21.npz")["b1"])
ncc = kdc.shape[2]
v = np.arange(NTV)
held = v[(v % 10) >= 8]                                   # 342 spokes neither method saw
print(f"(A) held-out spokes: {held.size} of {NTV}\n")

ctx = C.slice_ctx(21); rois = ctx["rois"]; body = ctx["BODY"]

def heldout_nmse(img):
    """forward-project a [192,192,nt] magnitude recon onto the held-out spokes and score."""
    nt = img.shape[-1]
    # embed the cropped image back onto the full grid the operator expects
    pad = (NGRID - img.shape[0]) // 2
    full = np.zeros((NGRID, NGRID, nt), np.float32)
    full[pad:pad+img.shape[0], pad:pad+img.shape[1]] = img
    # assign each held-out spoke to the frame whose window contains it
    fr = np.clip((held.astype(float) / NTV * nt).astype(int), 0, nt-1)
    tot_num = tot_den = 0.0
    for f in np.unique(fr):
        sp = held[fr == f]
        k = (Traj[:, sp] / NGRID).astype(np.complex128)[:, :, None]
        w = np.ones_like(np.abs(k))
        E = MCNUFFT(k, w, (b1/np.abs(b1).max()).astype(np.complex64), isign=ISIGN)
        pred = np.abs(E.forward(full[:, :, f][:, :, None].astype(np.complex128)))
        meas = np.abs(kdc[:, sp, :]).transpose(0, 1, 2)[:, :, :, None].squeeze(-1)
        p = pred.squeeze(-1).transpose(0, 1, 2) if pred.ndim == 4 else pred
        p = np.abs(p).ravel(); m = np.abs(meas).ravel()
        s = float((p @ m) / (p @ p + 1e-30))
        tot_num += float(np.sum((s*p - m)**2)); tot_den += float(np.sum(m**2))
    return float(np.sqrt(tot_num / (tot_den + 1e-30)))

def physical(img, label):
    """(B) reference-free bounds on the aorta first pass."""
    nt = img.shape[-1]; t = np.linspace(0, TA, nt)
    c = np.array([img[..., i][rois["aorta"]].mean() for i in range(nt)])
    c = c - np.median(c[:max(3, nt//40)])
    pk = c.max(); i = int(np.argmax(c)); half = pk/2
    l = i
    while l > 0 and c[l] > half: l -= 1
    r = i
    while r < nt-1 and c[r] > half: r += 1
    fwhm = t[r] - t[l]
    neg = float(np.mean(c < -0.05*pk))                      # fraction of clearly negative samples
    rise = c[:i+1]
    mono = float(np.mean(np.diff(rise) >= -0.02*pk)) if i > 1 else 1.0
    return dict(method=label, fwhm_s=float(fwhm), ttp_s=float(t[i]),
                neg_frac=neg, rise_monotone_frac=mono)

CASES = [("NIK k80 +heldout", f"{B}/results_sl21_k80/nik_slice_21.npy"),
         ("GRASP-v2 n12 k80", f"{GV}/gv2_slice21_n12_k80.npy"),
         ("GRASP-v2 n12 lam0.25", f"{GV}/gv2_slice21_n12.npy"),
         ("GRASP-v2 n12 lam0.02", f"{GV}/gv2_slice21_n12_lam0.02.npy")]
rows = []
for lab, p in CASES:
    if not os.path.exists(p): print(f"  {lab}: MISSING"); continue
    img = np.abs(np.load(p)).astype(np.float32)
    r = physical(img, lab); r["frames"] = int(img.shape[-1])
    try:
        r["heldout_nmse"] = heldout_nmse(img)
    except Exception as e:
        r["heldout_nmse"] = float("nan"); print(f"  {lab}: heldout failed {str(e)[:70]}")
    rows.append(r)

print(f"{'method':24} {'frames':>7} {'heldNMSE':>9} {'FWHM s':>8} {'ttp s':>7} {'neg%':>6} {'mono%':>6}")
for r in rows:
    print(f"{r['method']:24} {r['frames']:>7} {r['heldout_nmse']:>9.4f} {r['fwhm_s']:>8.1f} "
          f"{r['ttp_s']:>7.1f} {100*r['neg_frac']:>6.1f} {100*r['rise_monotone_frac']:>6.1f}")
# model-free reference for context only, NOT as the ruler
z = np.load(f"{B}/step2_slice21.npz"); mfimg = np.abs(z["mf"]).transpose(1,2,0).astype(np.float32)
rm = physical(mfimg, "model-free ref (context)")
print(f"{rm['method']:24} {mfimg.shape[-1]:>7} {'-':>9} {rm['fwhm_s']:>8.1f} {rm['ttp_s']:>7.1f} "
      f"{100*rm['neg_frac']:>6.1f} {100*rm['rise_monotone_frac']:>6.1f}")
json.dump(rows, open(f"{B}/v2_sweep_invivo/neutral_ruler.json", "w"), indent=1)
print("\nNEUTRAL_RULER_DONE")
