"""why did the k-space support prior leave the rendered air energy unchanged? render the coefficient images exactly as the trainer's prior does
(model.amplitudes on the nx cartesian grid, denormalize, disk mask, centred ifft2) and compare with the production render path
(nik_adapter.reconstruct_cartesian + recon_nik_cart, frame at 90 s): energy inside / outside the support mask on both, mask orientation candidates,
and a figure of both images with the mask contour. runs on the sup3 model and the base model.
usage: python support_diag.py --runs base:<dir>,sup3:<dir> --slice 21"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, argparse, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from types import SimpleNamespace
B = "/net/beegfs/users/P101440/DCE_NIK"; REFD = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"; sys.path.insert(0, B); sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
import dsp                                                                                   # dataset paths (DCE_DS=p3 default / p8 / p14)
import nik_adapter as A
from train_grasp_nik import build_model
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
from nik_output_recon import recon_nik_cart
import consolidated as C
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--slice", type=int, default=21); a = ap.parse_args(); Z = a.slice
    sh = A.load_shared(REFD); nx = int(sh["nx"]); bas = int(sh["bas"]); ds = A.make_radial_dataset(REFD, Z, compute_device=dev, shared=sh); b1 = ds["b1"]
    x, t, c, y_raw, sid = ds["x_all"], ds["t_all"], ds["coil_all"], ds["y_all_raw"], ds["spoke_id_all"]
    KEEP = np.load(dsp.KEEP); tr = torch.where(torch.isin(sid, torch.as_tensor(KEEP, device=dev, dtype=sid.dtype)))[0]
    dcf = compute_dcf_radial(x, method="simple_ramp"); nz = KSpaceNormalizer(); nz.fit(x[tr], y_raw[tr], dcf=dcf[tr], envelope_exponent=0.75)
    mask = np.load(f"{B}/spoke_masks/support_sl{Z}.npy").astype(bool); ctx = C.slice_ctx(Z); body = ctx["BODY"]; air = ctx["AIR"]; s0 = (nx - bas) // 2
    cg = torch.from_numpy(A.cartesian_grid(nx)).to(dev); disk = (torch.sqrt((cg ** 2).sum(1)) <= 1.0).float().view(nx, nx, 1).cpu().numpy()
    cands = {"id": mask, "rot180": mask[::-1, ::-1], "fliplr": mask[:, ::-1], "flipud": mask[::-1, :], "T": mask.T}
    rows = []; ims = []
    for spec in a.runs.split(","):
        nm, d = spec.split(":", 1); ck = torch.load(f"{d}/model_slice_{Z:02d}.pt", map_location=dev, weights_only=False)
        args = SimpleNamespace(**{k: ck[k] for k in ("model", "rank", "hidden", "depth", "w0", "s0", "coil_embed_dim", "k_freq", "k_sigma", "t_freq", "t_sigma", "ff_seed")},
                               patlak_free=0, aif_file=f"{B}/aif_slice{Z}.npz", tofts_basis=f"{B}/results/tofts_vs_patlak/basis_sl{Z}_r8_rms1.npz", phi_hidden=64, phi_depth=3, phi_w0=30.0, phi_ortho=False, n_pk=-1, radial_alpha=1.0, coil_mode=ck.get("coil_mode", "input"))
        m = build_model(args, int(ck["ncc"])).to(dev); m.load_state_dict(ck["state_dict"]); m.eval(); ncc = int(ck["ncc"]); Rk = int(m.rank)
        # the trainer's prior render
        E = np.zeros((nx, nx)); im0 = None
        with torch.no_grad():
            for cc in range(ncc):
                pr = []
                for i in range(0, cg.shape[0], 16384):
                    g = cg[i:i + 16384]; cd = torch.full((g.shape[0],), cc, dtype=torch.long, device=dev); Amp = m.amplitudes(g, cd)
                    pr.append(torch.stack([nz.denormalize(g, Amp[:, r, :]) for r in range(Rk)], 1))
                pr = torch.cat(pr, 0); K = (torch.complex(pr[..., 0], pr[..., 1]).view(nx, nx, Rk)).cpu().numpy() * disk
                im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(K, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)); E += (np.abs(im) ** 2).sum(-1)
                if im0 is None: im0 = np.abs(im[..., 0])
        fr = {k: float(E[mm].sum() / E.sum()) for k, mm in cands.items()}; pen = {k: float(E[~mm].mean() / (E[mm].mean() + 1e-12)) for k, mm in cands.items()}
        # the production render
        cart = A.reconstruct_cartesian(m, nz, REFD, device=dev.type, shared=sh, support_radius=1.0, verbose=False); img = recon_nik_cart(cart, b1, bas)
        i90 = img[..., int(np.argmin(np.abs(np.linspace(0, 375, img.shape[-1]) - 90)))]; airE = float(np.sqrt((i90[air] ** 2).mean()) / np.sqrt((i90[body] ** 2).mean()))
        crop_frac = float(E[s0:s0 + bas, s0:s0 + bas].sum() / E.sum()); outside_crop_body = float(E[s0:s0 + bas, s0:s0 + bas][~body].sum() / E[s0:s0 + bas, s0:s0 + bas].sum())
        rows.append((nm, fr, pen, airE, crop_frac, outside_crop_body)); ims.append((nm, im0, i90))
        print(f"{nm}: prior-render inside fraction {fr} | penalty (out/in mean) {pen} | energy in the {bas} crop {crop_frac:.3f}, of which outside body {outside_crop_body:.3f} | production render airE(90 s) {airE:.3f}", flush=True)
    fig, ax = plt.subplots(2, len(ims), figsize=(5 * len(ims), 9.5))
    for j, (nm, im0, i90) in enumerate(ims):
        ax[0, j].imshow(im0, cmap="gray", vmax=np.percentile(im0, 99.5)); ax[0, j].contour(mask.astype(float), levels=[0.5], colors=["lime"], linewidths=0.8); ax[0, j].set_title(f"{nm}: prior render |coef atom 0|, coil 0, {nx} grid, mask (lime)", fontsize=9); ax[0, j].axis("off")
        ax[1, j].imshow(i90, cmap="gray", vmax=np.percentile(i90[body], 99.5)); ax[1, j].contour(body.astype(float), levels=[0.5], colors=["lime"], linewidths=0.8); ax[1, j].contour(air.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.6); ax[1, j].set_title(f"{nm}: production render at 90 s, body (lime) / air (cyan)", fontsize=9); ax[1, j].axis("off")
    fig.tight_layout(); fig.savefig(f"{B}/results/tofts_vs_patlak/figures/support_diag_sl{Z}.png", dpi=120, facecolor="white"); print("SUPPORT_DIAG_DONE")

if __name__ == "__main__": main()
