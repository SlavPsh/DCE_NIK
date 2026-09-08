"""GRASP-Pro on the XCAT phantom via the SAME grasp_pro_py LIBRARY pathway as the real data:
GROG grid-to-Cartesian (self-calibrated) + Emat_GROG2Dksp + cs_l1_nlcg_sptv (NLCG). NO external solvers.
replaces the task4_gpu_recon direct-NUFFT + plain-FISTA path (blurred: no dcf, under-converged)."""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import precompute_ref as pr
import xph_pipeline as P, xph_common as X
K = 5; NITE = 5; NOUTER = 3

def build_phi(kdc, NLINE, NT, K=K):
    nx, nv, ncc = kdc.shape; c0 = nx // 2
    nav = np.abs(kdc[c0 - 2:c0 + 3, :NT * NLINE, :]).reshape(5, NLINE, NT, ncc, order="F").mean(1)
    ds = nav.transpose(0, 2, 1).reshape(5 * ncc, NT, order="F")
    w, PC = np.linalg.eigh(np.cov(ds, rowvar=False))
    return PC[:, np.argsort(-w)][:, :K].astype(np.complex64)

def reconstruct(zi=None):
    zi = P.ZI if zi is None else zi
    d = X.load_slice(zi); kx = d["kx"]; ky = d["ky"]; kdata = d["kdata"]          # (C,F,nang,RO)
    C, F, nang, RO = kdata.shape; tr = np.array(P.TRAIN_ANG); NLINE = len(tr); NT = F
    kdc = kdata[:, :, tr, :].transpose(3, 1, 2, 0).reshape(RO, F * NLINE, C)       # (RO, nv, C), col = f*NLINE+a
    traj = (kx[:, tr, :] + 1j * ky[:, tr, :]).transpose(2, 0, 1).reshape(RO, F * NLINE)  # (RO, nv), same order
    Traj_g = (traj * RO).astype(np.complex128)                                    # grid-index units [-RO/2,RO/2]
    kk = kdc[:, :, None, :]
    Gx, Gy = pr.get_gx_gy(kk, Traj_g)
    kref, _ = pr.grog_dictionary_interp(kk, Gx, Gy, Traj_g, 1)
    ref = np.squeeze(pr.ifft2c_mri(kref)); b1 = pr.adapt_array_2d(ref); b1 = (b1 / np.abs(b1).max()).astype(np.complex64)
    _, DCF = pr.grog_dictionary_interp(kk[:, -pr.Nqu:], Gx, Gy, Traj_g[:, -pr.Nqu:], 0)
    kdata2 = kdc.reshape(RO, NLINE, NT, C, order="F"); Traj2 = Traj_g.reshape(RO, NLINE, NT, order="F")
    kdata3, DCF_U = pr.grog_dictionary_interp(kdata2, Gx, Gy, Traj2, 1)            # [sx,sx,NT,C]
    sx = kdata3.shape[0]; print("GROG grid sx=%d (RO=%d), NLINE=%d NT=%d Nqu=%d" % (sx, RO, NLINE, NT, pr.Nqu), flush=True)
    mask = (kdata3[:, :, :, 0] != 0).astype(np.complex64)
    Weightc = np.repeat(DCF, NT, axis=2) / DCF_U
    Phi = build_phi(kdc, NLINE, NT); PCA = pr.TempPCASub(Phi)
    kdatac = np.stack([PCA @ kdata3[:, :, :, ii] for ii in range(C)], axis=3)
    Wc = PCA @ Weightc; Wc = np.repeat(Wc[:, :, :1], K, axis=2)[:, :, :, None]; Wc = np.repeat(Wc, C, axis=3)
    E = pr.Emat_GROG2Dksp(mask, b1, Wc, PCA, 1)
    y = (kdatac * np.sqrt(Wc)).astype(np.complex64)
    recon = E.H @ y
    param = dict(E=E, y=y, PCA=PCA, TV1=pr.TV_Temp(), TV2=pr.FD1OP(),
                 TVWeight1=np.abs(recon).max() * pr.Weight1, TVWeight2=np.abs(recon).max() * pr.Weight2, nite=NITE)
    for it in range(NOUTER):
        recon = pr.cs_l1_nlcg_sptv(recon, param); print("  nlcg outer %d/%d done" % (it + 1, NOUTER), flush=True)
    dyn = np.asarray(PCA.H @ recon)                                               # [sx,sx,NT] complex
    if sx != RO:                                                                  # match truth grid
        dyn = pr.crop_img(dyn, RO, RO) if sx > RO else dyn
    return dyn.astype(np.complex64), Phi
