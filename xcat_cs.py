"""CS (GRASP-Pro) on the XCAT sim: rank-K temporal subspace + spatial/temporal TV, via the
grasp GROG path adapted to XCAT geometry. XCAT-SPECIFIC (aligned stack, see xcat_adapter).
validate vs images.Recon, then extract aorta bolus for the B4 accuracy table.
usage: python xcat_cs.py [--slice 5]  -> results_xcat_cs/{cs_slice.npy, metrics.json}"""
import argparse, os, json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
import precompute_ref as pr
import xcat_adapter as X

K = pr.K   # 5


def build_phi(kdc, nline, nt):
    nx, nv, ncc = kdc.shape; c0 = nx // 2
    nav = np.abs(kdc[c0 - 2:c0 + 3, :nt * nline, :]).reshape(5, nline, nt, ncc, order="F").mean(1)  # (5,nt,ncc)
    ds = nav.transpose(0, 2, 1).reshape(5 * ncc, nt, order="F")
    w, PC = np.linalg.eigh(np.cov(ds, rowvar=False))
    return PC[:, np.argsort(-w)][:, :K].astype(np.complex64)


def recon(kdc, Traj_g, Phi, nline, nt, bas):
    nx, nv, ncc = kdc.shape
    kk = kdc[:, :, None, :]
    Gx, Gy = pr.get_gx_gy(kk, Traj_g)
    kref, _ = pr.grog_dictionary_interp(kk, Gx, Gy, Traj_g, 1)
    b1 = pr.adapt_array_2d(np.squeeze(pr.ifft2c_mri(kref))); b1 = (b1 / np.abs(b1).max()).astype(np.complex64)
    _, DCF = pr.grog_dictionary_interp(kk[:, -pr.Nqu:], Gx, Gy, Traj_g[:, -pr.Nqu:], 0)
    kdata2 = kdc[:, :nt * nline, :].reshape(nx, nline, nt, ncc, order="F")
    Traj2 = Traj_g[:, :nt * nline].reshape(nx, nline, nt, order="F")
    kdata3, DCF_U = pr.grog_dictionary_interp(kdata2, Gx, Gy, Traj2, 1)
    mask = (kdata3[:, :, :, 0] != 0).astype(np.complex64)
    Weightc = np.repeat(DCF, nt, axis=2) / DCF_U
    PCA = pr.TempPCASub(Phi)
    kdatac = np.stack([PCA @ kdata3[:, :, :, ii] for ii in range(ncc)], axis=3)
    Wc = PCA @ Weightc; Wc = np.repeat(Wc[:, :, :1], K, axis=2)[:, :, :, None]; Wc = np.repeat(Wc, ncc, axis=3)
    E = pr.Emat_GROG2Dksp(mask, b1, Wc, PCA, 1)
    y = (kdatac * np.sqrt(Wc)).astype(np.complex64)
    r = E.H @ y
    p = dict(E=E, y=y, PCA=PCA, TV1=pr.TV_Temp(), TV2=pr.FD1OP(),
             TVWeight1=np.abs(r).max() * pr.Weight1, TVWeight2=np.abs(r).max() * pr.Weight2, nite=5)
    for _ in range(3):
        r = pr.cs_l1_nlcg_sptv(r, p)
    return pr.crop_img(np.abs(PCA.H @ r), bas, bas).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, default=5); args = ap.parse_args()
    d = X.slice_radial(args.slice)
    kd = d["kdata"]; C, Fr, NA, RO = kd.shape                        # [C,F,9,RO]
    kdc = np.ascontiguousarray(kd.transpose(3, 1, 2, 0).reshape(RO, Fr * NA, C)).astype(np.complex64)  # frame-major
    Traj_g = np.ascontiguousarray((d["kx"] + 1j * d["ky"]).transpose(2, 0, 1).reshape(RO, Fr * NA)).astype(np.complex64)
    nline, nt, bas = NA, Fr, RO
    print(f"XCAT CS: kdc {kdc.shape}, nline {nline}, nt {nt}, bas {bas}", flush=True)
    Phi = build_phi(kdc, nline, nt)
    img = recon(kdc, Traj_g, Phi, nline, nt, bas)                   # [bas,bas,nt]
    os.makedirs("results_xcat_cs", exist_ok=True)
    np.save("results_xcat_cs/cs_slice.npy", img)
    # validate vs sim recon (temporal mean)
    ref = np.abs(d["ref"]).mean(0)                                  # [RO,152]
    s = (img.shape[1] - ref.shape[1]) // 2; myc = img[:, s:s + ref.shape[1]].mean(-1)
    corr = float(np.corrcoef((myc / myc.max()).ravel(), (ref / ref.max()).ravel())[0, 1])
    # aorta bolus accuracy
    roi = X.aorta_roi(d); tim = d["times"]
    cur = np.array([img[..., i][roi].mean() for i in range(nt)])
    b = cur[tim < 12].mean(); n = (cur - b) / (cur.max() - b + 1e-9)
    ttp = float(tim[np.argmax(n)]); half = (n > 0.5) & (tim < ttp + 40); fwhm = float(tim[half].max() - tim[half].min())
    tr = X.true_kinetics(zi=args.slice)
    res = dict(method="CS", corr_vs_simrecon=corr, ttp=ttp, fwhm=fwhm, ttp_true=tr["ttp"], fwhm_true=tr["fwhm"],
               ttp_err=ttp - tr["ttp"], fwhm_err=fwhm - tr["fwhm"])
    json.dump(res, open("results_xcat_cs/metrics.json", "w"), indent=1, default=float)
    print(f"CS geom corr {corr:.3f} | TTP {ttp:.1f}s (true {tr['ttp']:.1f}, err {ttp-tr['ttp']:+.1f}) | "
          f"FWHM {fwhm:.1f}s (true {tr['fwhm']:.1f}, err {fwhm-tr['fwhm']:+.1f})", flush=True)


if __name__ == "__main__":
    main()
