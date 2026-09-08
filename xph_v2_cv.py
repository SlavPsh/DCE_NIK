"""select grasp v2's lam by HELD-OUT cross-validation, not by looking at truth.

the lam sweep was scored against xcat truth. picking the best row from that is ORACLE selection and
would flatter grasp v2 exactly the way the temporal-PCA leak flattered grasp-pro. grasp-pro's K*=12
on this same phantom was chosen by held-out CV, so v2's lam must be chosen the same way.

VAL_ANG (angle 5) and TEST_ANG (angle 6) were never used for reconstruction. so for each recon we
forward-project onto the VAL spokes and measure data-consistency NMSE there. lowest val NMSE wins;
TEST is reported afterwards as the untouched check.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, glob
import numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK")
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_v2")
import xph_pipeline as P, xph_common as X
from grasp_v2_py import MCNUFFT

SW = f"{P.OUT}/v2_sweep"; A = f"{P.OUT}/arrays"; ISIGN = +1
z = np.load(f"{A}/xph_slice_cache.npz")
kx, ky, kdata, b1 = z["kx"], z["ky"], z["kdata"], z["b1"]
C, F, nang, RO = kdata.shape
b1n = (b1 / np.abs(b1).max()).astype(np.complex64)

def heldout_nmse(rec, G, ang):
    """forward-project rec onto held-out angle `ang` and compare to the measured spokes there."""
    nG = rec.shape[-1]
    kxs = kx[:nG*G, [ang], :].reshape(nG, G, RO); kys = ky[:nG*G, [ang], :].reshape(nG, G, RO)
    y   = kdata[:, :nG*G, ang, :].reshape(C, nG, G, RO)
    k = (kxs + 1j*kys).transpose(2, 1, 0)                       # [RO, G, nG]
    w = np.ones_like(np.abs(k))                                 # unweighted DC on held-out spokes
    E = MCNUFFT(k.astype(np.complex128), w.astype(np.float64), b1n, isign=ISIGN)
    pred = E.forward(rec.astype(np.complex128))                 # [RO, G, C, nG]
    meas = y.transpose(3, 2, 0, 1)                              # [RO, G, C, nG]
    # magnitude recon has no phase, so compare magnitudes with a global LS scale
    p = np.abs(pred).ravel(); m = np.abs(meas).ravel()
    s = float((p @ m) / (p @ p + 1e-30))
    return float(np.linalg.norm(s*p - m) / (np.linalg.norm(m) + 1e-30))

rows = []
for f in sorted(glob.glob(f"{SW}/v2_G*.npy")):
    bn = os.path.basename(f)[:-4]
    G = int(bn[4:6]); lam = float(bn.split("_lam")[1]) if "_lam" in bn else 0.25
    rec = np.load(f)
    j = f"{SW}/{bn}.json"
    tr = json.load(open(j)) if os.path.exists(j) else {}
    rows.append(dict(G=G, spf=5*G, lam=lam, frames=rec.shape[-1],
                     val=heldout_nmse(rec, G, P.VAL_ANG[0]), test=heldout_nmse(rec, G, P.TEST_ANG[0]),
                     haarpsi=tr.get("haarpsi"), c_aorta=tr.get("c_aorta"), aorta_pk=tr.get("aorta_pk")))
    print(f"  {bn}: val {rows[-1]['val']:.4f}", flush=True)

rows = [r for r in rows if r["spf"] >= 25]
rows.sort(key=lambda r: r["val"])
print(f"\nheld-out CV over grasp v2 (VAL = angle {P.VAL_ANG[0]}, never reconstructed)")
print(f"{'spf':>4} {'lam':>6} {'valNMSE':>8} {'testNMSE':>9} {'HaarPSI':>8} {'aortaC':>7} {'aortaPk':>8}")
for r in rows:
    print(f"{r['spf']:>4} {r['lam']:>6.2f} {r['val']:>8.4f} {r['test']:>9.4f} "
          f"{(r['haarpsi'] or 0):>8.4f} {(r['c_aorta'] or 0):>7.4f} {(r['aorta_pk'] or 0):>8.4f}")
best = rows[0]
print(f"\nCV-SELECTED (lowest val NMSE, truth never consulted): {best['spf']} spokes/frame, lam {best['lam']:g}")
print(f"  -> test NMSE {best['test']:.4f}, haarpsi {best['haarpsi']:.4f}, aortaC {best['c_aorta']:.4f}, peak {best['aorta_pk']:.4f}")
orc = min(rows, key=lambda r: r["c_aorta"] if r["c_aorta"] else 9)
print(f"  oracle-on-truth best curve would be {orc['spf']} spf lam {orc['lam']:g} (aortaC {orc['c_aorta']:.4f})"
      + ("  SAME as CV" if (orc['spf'], orc['lam']) == (best['spf'], best['lam']) else "  <- DIFFERENT, report the CV one"))
json.dump(rows, open(f"{SW}/lam_cv.json", "w"), indent=1)
print("CV_DONE")
