"""build + FREEZE the AIF-conditioned extended-Tofts temporal basis for nik_tofts_subspace.

input = the SAME AIF npz the Patlak model uses (normalized signal-domain enhancement on tC seconds),
so the Tofts arm has identical AIF information access.
  1. Patlak span [aif, iaif, 1] exactly as WIRE_FF_PATLAK builds it (peak-normalized), on the tC grid
  2. QR -> orthonormal Q, RECORDED R (coefficient change of coordinates: a_new = R a_old)
  3. ext-Tofts dictionary via dcenet_adapter.ext_tofts_sampled on the signal AIF as cp-proxy:
        C_t = vp*cp + Ktrans*int cp(tau-dt) exp(-kep (t-tau)),  Ktrans = ve*kep
     LINEAR SIGNAL PROXY (documented): the AIF is already signal-domain and max-normalized, so no
     SPGR is applied and no T1 is needed. the analytic-cp+SPGR variant is a separate check (--spgr).
  4. project dictionary OUT of span(Q), SVD, keep k extra atoms (rank-deficient sigma dropped)
  5. rank study on INDEPENDENT synthetic curves (other seed): first-pass / washout / total error
  6. save atoms + all transforms; evaluation on other grids = linear interp of the stored atoms
"""
import argparse, hashlib, json, os, sys
import numpy as np, torch
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import dcenet_adapter as D

PRIOR = dict(ke_min=0.05, ke_max=3.0,      # k_ep, min^-1, log-uniform. covers phantom pkLUT 0.42-1.66
             ve_min=0.02, ve_max=0.90,     # uniform. pkLUT 0-0.9
             vp_min=0.005, vp_max=0.60,    # uniform. pkLUT aorta 0.6
             dt_min_s=0.0, dt_max_s=20.0)  # arrival delay, seconds, uniform. pkLUT 0-15.6

def sample(n, seed):
    g = np.random.default_rng(seed)
    ke = np.exp(g.uniform(np.log(PRIOR["ke_min"]), np.log(PRIOR["ke_max"]), n))
    ve = g.uniform(PRIOR["ve_min"], PRIOR["ve_max"], n); vp = g.uniform(PRIOR["vp_min"], PRIOR["vp_max"], n)
    dt = g.uniform(PRIOR["dt_min_s"], PRIOR["dt_max_s"], n) / 60.0
    return ke, ve, vp, dt

def patlak_span(tC, aif):
    """identical to train_grasp_nik build_model + WIRE_FF_PATLAK.basis: aif/max, trapz-cumsum/max, ones"""
    a = aif / (aif.max() + 1e-9)
    ia = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(tC))]); ia = ia / (ia.max() + 1e-9)
    return np.stack([a, ia, np.ones_like(a)], 1)                                   # [G,3]

def dictionary(tC, aif, ke, ve, vp, dt, spgr=None, chunk=512):
    t_min = tC / 60.0; out = []
    with torch.no_grad():
        for i in range(0, ke.size, chunk):
            ct = D.ext_tofts_sampled(t_min, aif, ke[i:i+chunk], dt[i:i+chunk], ve[i:i+chunk], vp[i:i+chunk]).numpy()
            out.append(ct)
    Dm = np.concatenate(out, 0).astype(np.float64)                                  # [N,G] proxy enhancement
    if spgr is not None:                                                           # secondary variant only
        T1, TR, FA, r1 = spgr; R10 = 1000.0 / T1
        S = lambda R1: np.sin(np.deg2rad(FA)) * (1 - np.exp(-TR * R1 / 1000.0)) / (1 - np.cos(np.deg2rad(FA)) * np.exp(-TR * R1 / 1000.0))
        Dm = S(R10 + r1 * Dm) - S(R10)
    Dm = Dm / (np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12)                  # unit L2 per curve, shape only
    return Dm

def windows(tC, aif):
    tp = tC[int(np.argmax(aif))]
    return (tC <= tp + 20.0), (tC >= tp + 30.0)                                    # first pass / washout

def proj_err(B, X, m=None):
    """relative L2 error of projecting curves X [N,G] onto orthonormal columns B [G,R], optionally on mask m"""
    Xh = (X @ B) @ B.T
    if m is not None: X, Xh = X[:, m], Xh[:, m]
    return float(np.sqrt(((X - Xh) ** 2).sum() / ((X ** 2).sum() + 1e-30)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aif", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ranks", default="5,8,12,16"); ap.add_argument("--n-dict", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--thr-first", type=float, default=0.01); ap.add_argument("--thr-wash", type=float, default=0.01)
    ap.add_argument("--spgr", default=None, help="T1_ms,TR_ms,FA_deg,r1 -> analytic-signal variant (secondary check)")
    a = ap.parse_args()
    z = np.load(a.aif, allow_pickle=True); aif = np.asarray(z["aif_frame"], float); tC = np.asarray(z["tC"], float)
    aif_sha = hashlib.sha256(open(a.aif, "rb").read()).hexdigest()[:16]
    P = patlak_span(tC, aif); Q, R = np.linalg.qr(P)                               # Q [G,3] orthonormal, R [3,3]
    if np.any(np.diag(R) < 0): s = np.sign(np.diag(R)); Q = Q * s; R = (R.T * s).T  # fix sign convention
    spgr = tuple(float(x) for x in a.spgr.split(",")) if a.spgr else None
    ke, ve, vp, dt = sample(a.n_dict, a.seed); Dm = dictionary(tC, aif, ke, ve, vp, dt, spgr)
    Dp = Dm - (Dm @ Q) @ Q.T                                                       # project OUT of Patlak span
    U, S, Vt = np.linalg.svd(Dp.T, full_matrices=False)                            # U [G,k]: temporal atoms
    keep = S > 1e-6 * S[0]; U, S = U[:, keep], S[keep]                             # rank-deficient directions dropped
    ke2, ve2, vp2, dt2 = sample(a.n_test, a.seed + 1); Dtest = dictionary(tC, aif, ke2, ve2, vp2, dt2, spgr)
    fp, wo = windows(tC, aif); study = []
    for Rt in [int(x) for x in a.ranks.split(",")]:
        k = min(Rt - 3, U.shape[1]); B = np.concatenate([Q, U[:, :k]], 1)
        study.append(dict(rank=Rt, extras=k, err_total=proj_err(B, Dtest), err_firstpass=proj_err(B, Dtest, fp),
                          err_washout=proj_err(B, Dtest, wo), energy=float((S[:k] ** 2).sum() / (S ** 2).sum())))
    print(f"aif {os.path.basename(a.aif)} sha {aif_sha} | dict {a.n_dict} | test {a.n_test} | svd kept {U.shape[1]} | mode {'spgr' if spgr else 'linear-signal-proxy'}")
    print(f"{'rank':>4} {'extras':>6} {'total':>8} {'first':>8} {'wash':>8} {'energy':>7}")
    for s in study: print(f"{s['rank']:>4} {s['extras']:>6} {s['err_total']:>8.4f} {s['err_firstpass']:>8.4f} {s['err_washout']:>8.4f} {s['energy']:>7.4f}")
    ok = [s for s in study if s["err_firstpass"] <= a.thr_first and s["err_washout"] <= a.thr_wash]
    sel = min(ok, key=lambda s: s["rank"]) if ok else max(study, key=lambda s: s["rank"])
    print(f"SELECTED rank {sel['rank']} ({'meets' if ok else 'NONE meets'} first<={a.thr_first} wash<={a.thr_wash}; fallback=max)")
    k = sel["extras"]; atoms = np.concatenate([Q, U[:, :k]], 1).astype(np.float32)   # [G,R]
    np.savez(a.out, tgrid_s=tC.astype(np.float32), tgrid_model=(2 * tC / tC.max() - 1).astype(np.float32),
             atoms=atoms, R_patlak=R.astype(np.float32), Q_patlak=Q.astype(np.float32), sigma=S.astype(np.float32),
             rank=sel["rank"], n_fixed=3, prior=json.dumps(PRIOR), study=json.dumps(study), selected=json.dumps(sel),
             aif_file=a.aif, aif_sha256=aif_sha, mode="spgr" if spgr else "linear_signal_proxy", spgr=str(spgr),
             n_dict=a.n_dict, n_test=a.n_test, seed=a.seed, upstream_sha=D.upstream_sha())
    print("SAVED", a.out, "atoms", atoms.shape); print("BASIS_DONE")

if __name__ == "__main__": main()
