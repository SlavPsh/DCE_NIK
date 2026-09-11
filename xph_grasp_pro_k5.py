"""GRASP-Pro on the phantom through the library pathway (grog + Emat_GROG2Dksp + cs_l1_nlcg_sptv, the in vivo code),
with G consecutive frames binned (5G spokes/frame, F//G frames) and a chosen K. same 5 of 7 spokes as every other method.
out: arrays/grasp_pro_K<K>_G<G>.npz (rec oriented + one global scale to truth, Phi, orient, K, G, spf).
usage: XPH_SIM=nomotion python xph_grasp_pro_k5.py --K 5 --G 5"""
import warnings; warnings.filterwarnings("ignore")
import sys, argparse, time, numpy as np
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import precompute_ref as pr, xph_pipeline as P, xph_common as X, xph_grasp_lib as GL
from xph_v2_sweep import load_cached

def reconstruct(kx, ky, kdata, K, G, nite=GL.NITE, nouter=GL.NOUTER):
    C, F, nang, RO = kdata.shape; tr = np.array(P.TRAIN_ANG); nG = F // G; NLINE = len(tr) * G; NT = nG
    kdc = kdata[:, :nG * G][:, :, tr, :].transpose(3, 1, 2, 0).reshape(RO, nG * G * len(tr), C)      # (RO, nv, C), col = f*5 + a, frame-major
    traj = (kx[:nG * G][:, tr, :] + 1j * ky[:nG * G][:, tr, :]).transpose(2, 0, 1).reshape(RO, nG * G * len(tr))
    Traj_g = (traj * RO).astype(np.complex128)
    kk = kdc[:, :, None, :]; Gx, Gy = pr.get_gx_gy(kk, Traj_g)
    kref, _ = pr.grog_dictionary_interp(kk, Gx, Gy, Traj_g, 1); ref = np.squeeze(pr.ifft2c_mri(kref)); b1 = pr.adapt_array_2d(ref); b1 = (b1 / np.abs(b1).max()).astype(np.complex64)
    _, DCF = pr.grog_dictionary_interp(kk[:, -pr.Nqu:], Gx, Gy, Traj_g[:, -pr.Nqu:], 0)
    kdata2 = kdc.reshape(RO, NLINE, NT, C, order="F"); Traj2 = Traj_g.reshape(RO, NLINE, NT, order="F")   # G consecutive frames -> one bin
    kdata3, DCF_U = pr.grog_dictionary_interp(kdata2, Gx, Gy, Traj2, 1); sx = kdata3.shape[0]
    print(f"grog sx={sx} RO={RO} NLINE={NLINE} NT={NT} K={K}", flush=True)
    mask = (kdata3[:, :, :, 0] != 0).astype(np.complex64); Weightc = np.repeat(DCF, NT, axis=2) / DCF_U
    Phi = GL.build_phi(kdc, NLINE, NT, K=K); PCA = pr.TempPCASub(Phi)
    kdatac = np.stack([PCA @ kdata3[:, :, :, ii] for ii in range(C)], axis=3)
    Wc = PCA @ Weightc; Wc = np.repeat(Wc[:, :, :1], K, axis=2)[:, :, :, None]; Wc = np.repeat(Wc, C, axis=3)
    E = pr.Emat_GROG2Dksp(mask, b1, Wc, PCA, 1); y = (kdatac * np.sqrt(Wc)).astype(np.complex64); recon = E.H @ y
    param = dict(E=E, y=y, PCA=PCA, TV1=pr.TV_Temp(), TV2=pr.FD1OP(), TVWeight1=np.abs(recon).max() * pr.Weight1, TVWeight2=np.abs(recon).max() * pr.Weight2, nite=nite)
    for it in range(nouter): recon = pr.cs_l1_nlcg_sptv(recon, param); print(f"  nlcg outer {it + 1}/{nouter}", flush=True)
    dyn = np.asarray(PCA.H @ recon)
    if sx > RO: dyn = pr.crop_img(dyn, RO, RO)
    return dyn.astype(np.complex64), Phi, NLINE, NT

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=5); ap.add_argument("--G", type=int, default=5); a = ap.parse_args()
    t0 = time.time(); d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); A = f"{P.OUT}/arrays"
    kx, ky, kdata, _ = load_cached()
    dyn, Phi, NLINE, NT = reconstruct(kx, ky, kdata, a.K, a.G); rec0 = np.abs(dyn)
    Tw = np.stack([Tr[:, :, g * a.G:(g + 1) * a.G].mean(2) for g in range(NT)], -1); tm = Tw.mean(2)
    def corr(u, v): u = u[body].ravel() - u[body].mean(); v = v[body].ravel() - v[body].mean(); return float((u * v).sum() / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    def orient(v, nm): return {"id": v, "rot180": v[::-1, ::-1], "fliplr": v[:, ::-1], "flipud": v[::-1], "T": np.transpose(v, (1, 0, 2))}[nm]
    best = max(["id", "rot180", "fliplr", "flipud", "T"], key=lambda nm: corr(orient(rec0, nm).mean(2), tm)); rec = orient(rec0, best)
    s = np.sum(rec[body] * Tw[body]) / (np.sum(rec[body] ** 2) + 1e-12); rec = rec * s
    out = f"{A}/grasp_pro_K{a.K}_G{a.G}.npz"
    np.savez(out, rec=rec.astype(np.float32), Phi=Phi, orient=best, scale=float(s), K=a.K, G=a.G, spf=NLINE, frames=NT)
    nr = float(np.sqrt(np.mean((rec.mean(2)[body] - tm[body]) ** 2)) / (tm[body].max() - tm[body].min()))
    print(f"SAVED {out} rec {rec.shape} orient {best} corr {corr(rec.mean(2), tm):.3f} imgNRMSE(mean) {nr:.4f} ({time.time() - t0:.0f} s)"); print("GRASP_PRO_K5_DONE")

if __name__ == "__main__": main()
