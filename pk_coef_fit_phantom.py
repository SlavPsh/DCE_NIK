"""coefficient-domain parameter fit for the phantom nik-tofts model, vs truth and vs the image-domain fit.
the tofts model is k(x,y,t) = sum_r a_r(x,y) phi_r(t) with fixed orthonormal atoms phi (basis_xph.npz); path C gives the coefficient maps
a_r directly (nik_eval_*.npz: thC_best, one global scale). the fit here never renders frames: for every enhancing voxel the residual is
phi^T S(t; theta) - a, with S the exact spgr signal of the extended-kety curve (sim aif, label T10), theta = (ke, dt, ve, vp, S0).
phi orthonormal -> this equals a least-squares fit of the model to the voxel curve restricted to the basis span.
out: pk_maps/pk_coef_tofts.{npz,md}, figures/pk_coef_fit_phantom.png (truth | image-domain fit | coefficient-domain fit)
usage: XPH_SIM=nomotion python pk_coef_fit_phantom.py [--jobs 16] [--tag w768_ks2.5_s0_tofts16]"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, argparse, time, h5py, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from joblib import Parallel, delayed
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK"); sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK/third_party/DCENET")
import xph_pipeline as P, xph_common as X, DCE_matt as M
import pk_maps_phantom as PM

def spgr(T1_ms, TR_ms, fa_deg):
    E1 = np.exp(-TR_ms / T1_ms); a = np.deg2rad(fa_deg); return np.sin(a) * (1 - E1) / (1 - np.cos(a) * E1)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=16); ap.add_argument("--tag", default="w768_ks2.5_s0_tofts16"); ap.add_argument("--max-vox", type=int, default=0); a = ap.parse_args(); t0 = time.time()
    d = P.data(); tq = np.asarray(d["times"], float); lab = np.asarray(d["labels"]).astype(int); body = lab > 0; Tr = X.truth_at(P.ZI, tq).astype(np.float32); A = f"{P.OUT}/arrays"
    f = h5py.File(X.SIM, "r"); r = f["results"]; lut = {k: np.asarray(r["pkLUT"][k]).ravel() for k in ("ke", "ve", "vp", "dt")}
    TR = float(np.asarray(r["sim"]["TR"]).ravel()[0]); FA = float(np.asarray(r["sim"]["alpha"]).ravel()[0]); R1X = float(np.asarray(r["sim"]["relaxivity"]).ravel()[0]); TR_ms = TR * 1000 if TR < 1 else TR
    T1 = PM.t1_table(); T10 = np.full(lab.shape, np.nan); [np.putmask(T10, lab == l, T1.get(l, np.nan)) for l in np.unique(lab) if l > 0]
    ke_t, ve_t, vp_t = (np.where(body, lut[k][np.clip(lab - 1, 0, len(lut[k]) - 1)], np.nan) for k in ("ke", "ve", "vp"))
    enh = body & ((ke_t > 0) | (vp_t > 0)) & np.isfinite(T10) & (T10 > 0); idx = np.flatnonzero(enh.ravel())
    if a.max_vox and idx.size > a.max_vox: idx = np.random.default_rng(0).choice(idx, a.max_vox, replace=False)
    # coefficient maps (path C) in truth orientation, and the atoms on the time grid
    z = np.load(f"{A}/nik_eval_{a.tag}.npz", allow_pickle=True); thC = np.asarray(z["thC_best"]); s = float(z["scale_best"]); rec = np.asarray(z["rec_best"])
    rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1)); coef = np.stack([rot(thC[:, :, k]) for k in range(thC.shape[-1])], -1) * s     # [RO,RO,R] complex
    bz = np.load(f"{P.OUT}/../tofts_vs_patlak/basis_xph.npz", allow_pickle=True) if os.path.exists(f"{P.OUT}/../tofts_vs_patlak/basis_xph.npz") else np.load("/net/beegfs/users/P101440/DCE_NIK/results/tofts_vs_patlak/basis_xph.npz", allow_pickle=True)
    atoms, tg = np.asarray(bz["atoms"], float), np.asarray(bz["tgrid_s"], float); Rk = atoms.shape[1]
    Phi = np.stack([np.interp(tq, tg, atoms[:, k]) for k in range(Rk)], 1)                                                     # [T,R] on tq
    Q, _ = np.linalg.qr(Phi)                                                                                                       # orthonormal span(Phi) on tq
    chk = np.abs(np.einsum("xyr,tr->xyt", coef, Phi)); nr = float(np.linalg.norm((chk - rec)[body]) / (np.linalg.norm(rec[body]) + 1e-12))
    print(f"coef maps {coef.shape}, rank {Rk}, |coef @ Phi| vs rec_best nrmse {nr:.2e} (path C render check); voxels {idx.size}; TR {TR_ms:.2f} FA {FA:g} r1 {R1X:g}", flush=True)
    # per-voxel real coefficients: phase-align on the pre-contrast frames
    cv = coef.reshape(-1, Rk)[idx]; pre = tq < 10.0; s_pre = cv @ Phi[pre].T; ph = np.exp(-1j * np.angle(s_pre.mean(1))); av = np.real(cv * ph[:, None])
    T10v = T10.ravel()[idx]; tmin = tq / 60.0
    def model_coef(theta, T10_ms):
        ke, dt, ve, vp, S0 = theta; Ct = M.Cosine4AIF_ExtKety(tmin, PM.AIF, ke, dt, ve, vp)
        S = S0 * spgr(1000.0 / (1000.0 / T10_ms + R1X * np.clip(Ct, 0, None)), TR_ms, FA) / spgr(T10_ms, TR_ms, FA)
        return Q.T @ S                                                                                                          # coefficients in the orthonormalized span
    def fit_one(i):
        target = Q.T @ (Phi @ av[i]); S0 = float((Phi @ av[i])[pre].mean()); T10_ms = float(T10v[i])
        best = None
        for x0 in ((1.0, 0.03, 0.3, 0.05, S0), (0.3, 0.1, 0.6, 0.02, S0)):
            try:
                res = least_squares(lambda th: model_coef(th, T10_ms) - target, x0, bounds=([1e-3, 0.0, 0.01, 0.0, 0.0], [5.0, 0.5, 1.0, 1.0, np.inf]), max_nfev=200)
                if best is None or res.cost < best.cost: best = res
            except Exception: pass
        return best.x if best is not None else np.full(5, np.nan)
    ts = time.time(); par = np.array(Parallel(n_jobs=a.jobs)(delayed(fit_one)(i) for i in range(idx.size))); print(f"coefficient-domain fit: {idx.size} voxels in {time.time() - ts:.0f} s", flush=True)
    maps = {}
    for j, nm in enumerate(("ke", "dt", "ve", "vp", "S0")):
        m = np.full(lab.size, np.nan); m[idx] = par[:, j]; maps[nm] = m.reshape(lab.shape)
    maps["ktrans"] = maps["ke"] * maps["ve"]
    zi = np.load(f"{P.OUT}/pk_maps/pk_maps.npz"); img = {nm: zi[f"tofts_{nm}"] for nm in ("ke", "ve", "vp", "ktrans")}; truth = {nm: zi[f"truth_lut_{nm}"] for nm in ("ke", "ve", "vp", "ktrans")}
    labs = [int(l) for l in np.unique(lab) if l > 0 and enh[lab == l].sum() > 30]
    rows = {l: {nm: [float(np.nanmedian(truth[nm][(lab == l) & enh])), float(np.nanmedian(img[nm][(lab == l) & enh])), float(np.nanmedian(maps[nm][(lab == l) & enh]))] for nm in ("ktrans", "ve", "vp", "ke")} for l in labs}
    md = [f"# nik-tofts on the phantom: coefficient-domain fit vs image-domain fit vs truth (per-label medians over {idx.size} enhancing voxels; tag {a.tag})", "",
          "coefficient-domain: residual phi^T S(t; ke, dt, ve, vp, S0) - a in the orthonormalized span, exact spgr with the label T10, sim aif, no frames rendered; image-domain: the DCE-NET fit of pk_maps_phantom.py on the rendered series", ""]
    for nm in ("ktrans", "ve", "vp", "ke"):
        md += [f"## {nm}", "| label | truth | image-domain fit | coefficient-domain fit |", "|---|---|---|---|"] + [f"| {l} | {rows[l][nm][0]:.3f} | {rows[l][nm][1]:.3f} | {rows[l][nm][2]:.3f} |" for l in labs] + [""]
    err = {nm: {k: float(np.nanmedian(np.abs(v[enh] - truth[nm][enh]) / (np.abs(truth[nm][enh]) + 1e-6))) for k, v in (("image", img[nm]), ("coef", maps[nm]))} for nm in ("ktrans", "ve", "vp")}
    md += ["## per-voxel median relative error vs truth (enhancing voxels)", "| param | image-domain | coefficient-domain |", "|---|---|---|"] + [f"| {nm} | {err[nm]['image']:.3f} | {err[nm]['coef']:.3f} |" for nm in err]
    os.makedirs(f"{P.OUT}/pk_maps", exist_ok=True); open(f"{P.OUT}/pk_maps/pk_coef_tofts.md", "w").write("\n".join(md)); print("\n".join(md))
    np.savez(f"{P.OUT}/pk_maps/pk_coef_tofts.npz", enh=enh, **{f"coef_{nm}": maps[nm] for nm in maps}); json.dump(dict(rows=rows, err=err, render_check_nrmse=nr), open(f"{P.OUT}/pk_maps/pk_coef_tofts.json", "w"), indent=1)
    cols = [("truth (pkLUT)", truth), ("NIK-tofts, image-domain fit", img), ("NIK-tofts, coefficient-domain fit", maps)]
    fig, ax = plt.subplots(3, 3, figsize=(9.6, 9.2))
    for j, (nm_c, mp) in enumerate(cols):
        for i, (nm, vmax) in enumerate((("ktrans", np.nanpercentile(truth["ktrans"], 99)), ("ve", 1.0), ("vp", np.nanpercentile(truth["vp"], 99)))):
            ax[i, j].imshow(np.nan_to_num(np.where(enh, mp[nm], np.nan)), cmap="inferno" if nm == "ktrans" else ("viridis" if nm == "vp" else "magma"), vmin=0, vmax=max(vmax, 1e-3)); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(nm_c, fontsize=11, fontweight="bold")
            if j == 0: ax[i, j].text(-0.06, 0.5, {"ktrans": "Ktrans (1/min)", "ve": "ve", "vp": "vp"}[nm], transform=ax[i, j].transAxes, rotation=90, va="center", fontsize=12)
    fig.suptitle("nik-tofts, phantom: parameters from the rendered images vs from the coefficients directly", fontsize=12); fig.tight_layout()
    fig.savefig(f"{P.OUT}/figures/pk_coef_fit_phantom.png", dpi=150, facecolor="white"); print(f"saved {P.OUT}/figures/pk_coef_fit_phantom.png ({time.time() - t0:.0f} s)"); print("PK_COEF_DONE")

if __name__ == "__main__": main()
