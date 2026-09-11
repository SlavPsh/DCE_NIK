"""GRASP-Pro on the phantom through the NUFFT pathway of the report (xph_grasp_nufft.Emat_NUFFT + cs_l1_nlcg_sptv, the code behind
arrays/grasp_recon.npz), with G consecutive frames binned (5G spokes/frame, F//G frames) and a chosen K. same 5 of 7 spokes.
out: arrays/grasp_pro_K<K>_G<G>.npz (rec oriented + one global scale to the window-averaged truth, Phi, orient, K, G, spf, frames).
usage: XPH_SIM=nomotion python xph_grasp_pro_k5.py --K 5 --G 5   (gpu: cufinufft)"""
import warnings; warnings.filterwarnings("ignore")
import sys, argparse, time, numpy as np
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
import precompute_ref as pr, xph_pipeline as P, xph_common as X, xph_grasp_nufft as GN

def build_phi_binned(kdata, tr, G, K):
    C, F, nang, RO = kdata.shape; c0 = RO // 2; nG = F // G
    nav = np.abs(kdata[:, :nG * G, tr, c0 - 2:c0 + 3]).reshape(C, nG, G, len(tr), 5).mean(2)      # bin = mean over its G frames
    ds = nav.transpose(0, 2, 3, 1).reshape(-1, nG)
    w, PC = np.linalg.eigh(np.cov(ds, rowvar=False)); return PC[:, np.argsort(-w)][:, :K].astype(np.complex64)

def reconstruct(d, K, G):
    kx, ky, kdata, b1 = d["kx"], d["ky"], d["kdata"], d["b1"]; C, F, nang, RO = kdata.shape; tr = np.array(P.TRAIN_ANG); nG = F // G
    fr = lambda g: slice(g * G, (g + 1) * G)
    trajs = [(-kx[fr(g)][:, tr].reshape(-1, RO), -ky[fr(g)][:, tr].reshape(-1, RO)) for g in range(nG)]            # (G*spf, RO) per bin
    dcf = [np.maximum(np.abs(kx[fr(g)][:, tr] + 1j * ky[fr(g)][:, tr]).reshape(-1, RO), 1e-3) for g in range(nG)]
    Phi = build_phi_binned(kdata, tr, G, K); PCA = pr.TempPCASub(Phi)
    E = GN.Emat_NUFFT(trajs, dcf, b1, Phi, RO)
    raw = np.stack([kdata[:, fr(g)][:, :, tr, :].reshape(C, -1) for g in range(nG)], 1).astype(np.complex128)     # same (G, spf, RO) order as trajs
    y = E.apply_dcf(raw); recon = E.H @ y
    print(f"K={K} G={G}: {nG} bins of {G * len(tr)} spokes | op-scale a={E.a:.3e} | |recon|max {np.abs(recon).max():.3e}", flush=True)
    param = dict(E=E, y=y, PCA=PCA, TV1=pr.TV_Temp(), TV2=pr.FD1OP(), TVWeight1=np.abs(recon).max() * pr.Weight1, TVWeight2=np.abs(recon).max() * pr.Weight2, nite=GN.NITE)
    for it in range(GN.NOUTER):
        recon = pr.cs_l1_nlcg_sptv(recon, param); print(f"  nlcg outer {it + 1}/{GN.NOUTER} finite={np.isfinite(recon).all()}", flush=True)
    return np.asarray(PCA.H @ recon).astype(np.complex64), Phi, nG, G * len(tr)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=5); ap.add_argument("--G", type=int, default=5); a = ap.parse_args()
    t0 = time.time(); d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); A = f"{P.OUT}/arrays"
    dyn, Phi, NT, spf = reconstruct(d, a.K, a.G); rec0 = np.abs(dyn)
    Tw = np.stack([Tr[:, :, g * a.G:(g + 1) * a.G].mean(2) for g in range(NT)], -1); tm = Tw.mean(2)
    def corr(u, v): u = u[body].ravel() - u[body].mean(); v = v[body].ravel() - v[body].mean(); return float((u * v).sum() / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    def orient(v, nm): return {"id": v, "rot180": v[::-1, ::-1], "fliplr": v[:, ::-1], "flipud": v[::-1], "T": np.transpose(v, (1, 0, 2)), "T_rot180": np.transpose(v, (1, 0, 2))[::-1, ::-1]}[nm]
    best = max(["id", "rot180", "fliplr", "flipud", "T", "T_rot180"], key=lambda nm: corr(orient(rec0, nm).mean(2), tm)); rec = orient(rec0, best)
    s = np.sum(rec[body] * Tw[body]) / (np.sum(rec[body] ** 2) + 1e-12); rec = rec * s
    out = f"{A}/grasp_pro_K{a.K}_G{a.G}.npz"
    np.savez(out, rec=rec.astype(np.float32), Phi=Phi, orient=best, scale=float(s), K=a.K, G=a.G, spf=spf, frames=NT)
    nr = float(np.sqrt(np.mean((rec.mean(2)[body] - tm[body]) ** 2)) / (tm[body].max() - tm[body].min()))
    print(f"SAVED {out} rec {rec.shape} orient {best} corr {corr(rec.mean(2), tm):.3f} imgNRMSE(mean) {nr:.4f} ({time.time() - t0:.0f} s)"); print("GRASP_PRO_K5_DONE")

if __name__ == "__main__": main()
