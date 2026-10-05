"""artifacts at contrast arrival (user observation 2026-10-05): three readouts on the existing recons and models, no retraining.
(1) time-resolved artifact metrics per arm on the arm's own frame grid: air energy (rms in air / rms in body of the frame), fine-scale fraction
    (high-pass sigma 3 rms / rms, body), temporal roughness (frame-to-frame rms difference in the non-enhancing body / mean); summarized in
    windows pre (t < 40), arrival (arrival-5 .. arrival+20), plateau (90 to 150), late (> 200). spike at arrival in the nik arms = the question.
(2) per-atom analysis of the tofts8 (or patlak) model: the coefficient images per atom (coil combined with the saved b1), their air energy,
    their share of the image at arrival, and the effective spoke count of each atom n_eff = (sum_s phi_r(t_s)^2)^2 / sum_s phi_r(t_s)^4 over
    the kept spokes (how many spokes constrain that map). fast atoms with few spokes and high air energy = the undersampling mechanism.
(3) k-space residual of the model per spoke (normalized units as in the training loss, 1 in RO_SUB readout points), binned in time: a spike at
    arrival = the atoms cannot follow the data there (model mismatch), flat = the data are fitted but the maps are streaky (regularization).
usage: DCE_DS=p3 python arrival_artifact_diag.py --slice 21 --items "label:path,..." --model "tofts8:<run dir>" --tag x"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, argparse, numpy as np, torch, scipy.ndimage as ndi
from types import SimpleNamespace
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B); sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
import dsp, consolidated as C, nik_adapter as A
from train_grasp_nik import build_model
from kspace_normalization import KSpaceNormalizer, compute_dcf_radial
from nik_output_recon import recon_nik_cart

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); ap.add_argument("--items", required=True); ap.add_argument("--model", default=""); ap.add_argument("--tag", default="")
    ap.add_argument("--ro-sub", type=int, default=8); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); a = ap.parse_args(); Z = a.slice; TA = dsp.TA; dev = torch.device(a.device)
    ctx = C.slice_ctx(Z); body = ctx["BODY"]; air = ctx["AIR"]; nonenh = ctx["NONENH"]; rois = ctx["rois"]
    z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float)
    az = np.load(dsp.AIF(Z)); aif = np.asarray(az["aif_frame"], float); tC = np.asarray(az["tC"], float)
    t_arr = float(tC[np.argmax(aif > 0.15 * aif.max())]); t_pk = float(tC[np.argmax(aif)])
    WIN = {"pre": (0.0, 40.0), "arrival": (t_arr - 5.0, t_arr + 20.0), "plateau": (90.0, 150.0), "late": (200.0, TA)}
    out = f"{B}/results/tofts_vs_patlak/arrival_artifact{dsp.SFX}_sl{Z}{a.tag}"; R = {"slice": Z, "t_arrival": t_arr, "t_peak": t_pk, "windows": WIN}
    # (1) time-resolved metrics
    def hp(x): return x - ndi.gaussian_filter(x, 3.0)
    def metrics(v, t):
        airE = np.array([np.sqrt((v[..., i][air] ** 2).mean()) / (np.sqrt((v[..., i][body] ** 2).mean()) + 1e-12) for i in range(v.shape[-1])])
        fine = np.array([np.sqrt((hp(v[..., i])[body] ** 2).mean()) / (np.sqrt((v[..., i][body] ** 2).mean()) + 1e-12) for i in range(v.shape[-1])])
        d = np.diff(v, axis=-1); rough = np.array([np.sqrt((d[..., i][nonenh] ** 2).mean()) / (np.abs(v[..., i][nonenh]).mean() + 1e-12) for i in range(d.shape[-1])])
        return airE, fine, np.concatenate([[rough[0]], rough])
    arms = [("model-free 31-spoke", mf, tmf)]
    for spec in [x for x in a.items.split(",") if x]:
        lab, path = spec.split(":", 1)
        if not os.path.exists(path): print("missing", path); continue
        v = np.abs(np.load(path)).astype(np.float32); assert v.shape[:2] == body.shape, (lab, v.shape); arms.append((lab, v, (np.arange(v.shape[-1]) + 0.5) * TA / v.shape[-1]))
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True); T1 = {}
    for lab, v, t in arms:
        airE, fine, rough = metrics(v, t); T1[lab] = {}
        for k, (lo, hi) in WIN.items():
            m = (t >= lo) & (t < hi); T1[lab][k] = dict(airE=float(airE[m].mean()), fine=float(fine[m].mean()), rough=float(rough[m].mean())) if m.any() else {}
        for k, (y, nm) in enumerate(((airE, "air energy (rms air / rms body)"), (fine, "fine-scale fraction (hp sigma 3 / rms, body)"), (rough, "temporal roughness (non-enhancing body)"))):
            ax[k].plot(t, y, lw=0.9, label=lab); ax[k].set_ylabel(nm, fontsize=8)
    for k in range(3): ax[k].axvspan(WIN["arrival"][0], WIN["arrival"][1], color="orange", alpha=0.12); ax[k].axvline(t_pk, color="r", lw=0.6)
    ax[0].legend(fontsize=7, ncol=4); ax[-1].set_xlabel("t [s]"); fig.suptitle(f"{dsp.DS} slice {Z}: artifact level vs time (orange = arrival window, red = aif peak)", fontsize=10); fig.tight_layout(); fig.savefig(out + "_time.png", dpi=120, facecolor="white")
    R["time_windows"] = T1
    # (2) + (3) model
    if a.model:
        nm, d = a.model.split(":", 1); ck = torch.load(f"{d}/model_slice_{Z:02d}.pt", map_location="cpu", weights_only=False)
        args = SimpleNamespace(**{k: ck[k] for k in ("model", "rank", "hidden", "depth", "w0", "s0", "coil_embed_dim", "k_freq", "k_sigma", "t_freq", "t_sigma", "ff_seed")}, patlak_free=0,
                               aif_file=dsp.AIF(Z), tofts_basis=dsp.BASIS(Z, 8), phi_hidden=ck.get("phi_hidden", 64), phi_depth=ck.get("phi_depth", 3), phi_w0=ck.get("phi_w0", 30.0), phi_ortho=ck.get("phi_ortho", False), n_pk=-1, radial_alpha=1.0, coil_mode=ck.get("coil_mode", "input"))
        m = build_model(args, int(ck["ncc"])); m.load_state_dict(ck["state_dict"]); m = m.to(dev).eval(); ncc = int(ck["ncc"]); Rk = int(m.rank)
        sh = A.load_shared(dsp.REF); ds = A.make_radial_dataset(dsp.REF, Z, compute_device="cpu", shared=sh)
        x, t, c, y_raw, sid, ro = ds["x_all"], ds["t_all"], ds["coil_all"], ds["y_all_raw"], ds["spoke_id_all"], ds["ro_id_all"]
        KEEP = np.load(dsp.KEEP); keep_t = torch.as_tensor(KEEP, dtype=sid.dtype); tr = torch.where(torch.isin(sid, keep_t))[0]
        dcf = compute_dcf_radial(x, method="simple_ramp"); nz = KSpaceNormalizer(); nz.fit(x[tr], y_raw[tr], dcf=dcf[tr], envelope_exponent=0.75); y = nz.normalize(x, y_raw)
        sl = np.load(f"{dsp.REF}/slice_{Z:02d}.npz"); b1 = np.asarray(sl["b1"]).astype(np.complex64); nx = int(sh["nx"]); bas = int(sh["bas"])
        # per-atom coefficient images
        grid = torch.from_numpy(A.cartesian_grid(nx)).to(dev); cart = np.zeros((nx, nx, Rk, ncc), np.complex64)
        with torch.no_grad():
            for cc in range(ncc):
                cd = torch.full((grid.shape[0],), cc, dtype=torch.long, device=dev); Amp = torch.cat([m.amplitudes(grid[i:i + 65536], cd[i:i + 65536]) for i in range(0, grid.shape[0], 65536)], 0)
                for r in range(Rk):
                    pr = nz.denormalize(grid, Amp[:, r, :]).cpu().numpy(); cart[:, :, r, cc] = (pr[:, 0] + 1j * pr[:, 1]).reshape(nx, nx)
        rr = np.sqrt((A.cartesian_grid(nx) ** 2).sum(1)).reshape(nx, nx); cart[rr > 1.0] = 0
        _, atom_img = recon_nik_cart(cart, b1, bas, return_complex=True)                      # [bas,bas,R] complex coefficient images
        tg = torch.linspace(-1, 1, 2048);
        with torch.no_grad(): P = m.basis(tg.to(dev)).cpu().numpy(); phi = P[..., 0] + 1j * P[..., 1]           # [2048,R]
        with torch.no_grad(): Ps = m.basis(t[tr].to(dev)).cpu().numpy(); phis = np.abs(Ps[..., 0] + 1j * Ps[..., 1]) ** 2   # [N_tr,R] at the kept samples
        # per spoke (not per sample): one value per kept spoke
        sp = sid[tr].numpy(); usp, inv = np.unique(sp, return_inverse=True); phi_sp = np.zeros((len(usp), Rk)); cnt = np.bincount(inv)
        for r in range(Rk): phi_sp[:, r] = np.bincount(inv, weights=phis[:, r]) / cnt
        n_eff = (phi_sp.sum(0) ** 2) / ((phi_sp ** 2).sum(0) + 1e-12)
        ts_all = (2.0 * np.asarray(sh["view_time"]).ravel() - 1.0); ts_model = np.asarray(sh["view_time"]).ravel() * TA
        i_arr = int(np.argmin(np.abs((tg.numpy() + 1) / 2 * TA - (t_arr + 8.0)))); i_late = int(np.argmin(np.abs((tg.numpy() + 1) / 2 * TA - 300.0)))
        atoms = []
        for r in range(Rk):
            im = np.abs(atom_img[..., r]); airE = float(np.sqrt((im[air] ** 2).mean()) / (np.sqrt((im[body] ** 2).mean()) + 1e-12))
            fine = float(np.sqrt((hp(im)[body] ** 2).mean()) / (np.sqrt((im[body] ** 2).mean()) + 1e-12))
            contrib_arr = float(np.abs(phi[i_arr, r]) * np.sqrt((im[body] ** 2).mean())); contrib_late = float(np.abs(phi[i_late, r]) * np.sqrt((im[body] ** 2).mean()))
            atoms.append(dict(atom=r, n_eff_spokes=float(n_eff[r]), air_energy=airE, fine_fraction=fine, rms_body=float(np.sqrt((im[body] ** 2).mean())), contrib_arrival=contrib_arr, contrib_late=contrib_late))
        tot_arr = sum(d_["contrib_arrival"] for d_ in atoms) + 1e-12; tot_late = sum(d_["contrib_late"] for d_ in atoms) + 1e-12
        for d_ in atoms: d_["share_arrival"] = d_["contrib_arrival"] / tot_arr; d_["share_late"] = d_["contrib_late"] / tot_late
        R["atoms"] = atoms; R["model"] = nm
        # figure: atom images + temporal profiles
        fig, ax = plt.subplots(3, Rk, figsize=(2.4 * Rk, 7.2))
        for r in range(Rk):
            im = np.abs(atom_img[..., r]); ax[0, r].imshow(im, cmap="gray", vmin=0, vmax=np.percentile(im[body], 99.5)); ax[0, r].axis("off"); ax[0, r].set_title(f"atom {r}\nair {atoms[r]['air_energy']:.2f} n_eff {n_eff[r]:.0f}", fontsize=8)
            ax[1, r].imshow(np.clip(im / (np.percentile(im[body], 99.5) + 1e-12), 0, 0.25), cmap="gray"); ax[1, r].axis("off"); ax[1, r].set_title("x4 (air / streaks)", fontsize=7)
            ax[2, r].plot((tg.numpy() + 1) / 2 * TA, np.real(phi[:, r]), lw=0.8); ax[2, r].axvspan(WIN["arrival"][0], WIN["arrival"][1], color="orange", alpha=0.15); ax[2, r].set_title(f"share arrival {atoms[r]['share_arrival']:.2f} late {atoms[r]['share_late']:.2f}", fontsize=7); ax[2, r].tick_params(labelsize=6)
        fig.suptitle(f"{dsp.DS} slice {Z}, {nm}: coefficient images per atom (coil combined), air energy, effective spoke count, temporal profile", fontsize=10); fig.tight_layout(); fig.savefig(out + "_atoms.png", dpi=110, facecolor="white")
        # (3) residual per spoke vs time (subsampled readouts)
        sub = tr[(ro[tr] % a.ro_sub) == 0]; res = np.zeros(len(usp)); en = np.zeros(len(usp)); cnt2 = np.zeros(len(usp))
        with torch.no_grad():
            for i in range(0, sub.numel(), 262144):
                j = sub[i:i + 262144]; p = m(x[j].to(dev), t[j].to(dev), c[j].to(dev)).cpu(); e = ((p - y[j]) ** 2).sum(1).numpy(); yy = (y[j] ** 2).sum(1).numpy()
                k = np.searchsorted(usp, sid[j].numpy()); np.add.at(res, k, e); np.add.at(en, k, yy); np.add.at(cnt2, k, 1)
        nmse_sp = res / (en + 1e-12); mse_sp = res / np.maximum(cnt2, 1); tsp = ts_model[usp]
        fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True); o = np.argsort(tsp)
        k9 = 9; sm = lambda v: np.convolve(np.pad(v, (k9 // 2, k9 // 2), mode="edge"), np.ones(k9) / k9, mode="valid")
        ax[0].plot(tsp[o], nmse_sp[o], lw=0.4, alpha=0.5); ax[0].plot(tsp[o], sm(nmse_sp[o]), lw=1.2, color="k"); ax[0].set_ylabel("k-space NMSE per spoke"); ax[0].axvspan(WIN["arrival"][0], WIN["arrival"][1], color="orange", alpha=0.15)
        ax[1].plot(tsp[o], mse_sp[o], lw=0.4, alpha=0.5); ax[1].plot(tsp[o], sm(mse_sp[o]), lw=1.2, color="k"); ax[1].set_ylabel("k-space MSE per spoke (normalized units)"); ax[1].axvspan(WIN["arrival"][0], WIN["arrival"][1], color="orange", alpha=0.15); ax[1].set_xlabel("spoke time [s]")
        fig.suptitle(f"{dsp.DS} slice {Z}, {nm}: training residual per kept spoke vs time (1 in {a.ro_sub} readout points)", fontsize=10); fig.tight_layout(); fig.savefig(out + "_residual.png", dpi=120, facecolor="white")
        RW = {}
        for k, (lo, hi) in WIN.items():
            mm = (tsp >= lo) & (tsp < hi); RW[k] = dict(nmse=float(nmse_sp[mm].mean()), mse=float(mse_sp[mm].mean()), n_spokes=int(mm.sum())) if mm.any() else {}
        R["residual_windows"] = RW
    json.dump(R, open(out + ".json", "w"), indent=1)
    L = [f"# artifacts at contrast arrival, {dsp.DS} slice {Z}; arrival {t_arr:.0f} s, aif peak {t_pk:.0f} s; windows " + ", ".join(f"{k} {lo:.0f}-{hi:.0f} s" for k, (lo, hi) in WIN.items()), "",
         "## (1) artifact level per window (air energy / fine-scale fraction / temporal roughness)", "", "| arm | " + " | ".join(WIN) + " |", "|---|" + "---|" * len(WIN)]
    for lab in T1: L.append(f"| {lab} | " + " | ".join(f"{T1[lab][k]['airE']:.3f} / {T1[lab][k]['fine']:.3f} / {T1[lab][k]['rough']:.3f}" if T1[lab][k] else "-" for k in WIN) + " |")
    if a.model:
        L += ["", f"## (2) {nm}: per-atom coefficient images", "", "| atom | n_eff spokes | air energy | fine fraction | rms body | share at arrival | share late |", "|---|---|---|---|---|---|---|"]
        for d_ in R["atoms"]: L.append(f"| {d_['atom']} | {d_['n_eff_spokes']:.0f} | {d_['air_energy']:.3f} | {d_['fine_fraction']:.3f} | {d_['rms_body']:.3g} | {d_['share_arrival']:.2f} | {d_['share_late']:.2f} |")
        L += ["", f"## (3) {nm}: k-space residual per spoke by window", "", "| window | spokes | NMSE | MSE (normalized) |", "|---|---|---|---|"]
        for k, v in R["residual_windows"].items(): L.append(f"| {k} | {v.get('n_spokes', 0)} | {v.get('nmse', float('nan')):.4f} | {v.get('mse', float('nan')):.4g} |")
    open(out + ".md", "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("ARRIVAL_DONE")

if __name__ == "__main__": main()
