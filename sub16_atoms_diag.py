"""sub16 atom diagnostic: for every run dir, rebuild the model from model_slice_<Z>.pt, evaluate the 16 learned atoms on a dense time grid, and
measure their respiratory content (power above 0.1 Hz / total, after removing the mean) next to the oscillation of the roi curves of the saved
recon (high-pass temporal std / mean, 9-frame running mean removed) and the curve nrmse vs model-free. atoms figure per run. the tofts8 atoms
(fixed) and the production sub16 are the two references. usage: python sub16_atoms_diag.py --slice 21 --runs "label:dir,..." --tag x"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, argparse, numpy as np, torch
from types import SimpleNamespace
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp, consolidated as C
from train_grasp_nik import build_model

def hp_std(c): return float(np.std((c - np.convolve(c, np.ones(9) / 9, mode="same"))[5:-5]) / (np.abs(c).mean() + 1e-12))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, required=True); ap.add_argument("--runs", required=True); ap.add_argument("--tag", default=""); a = ap.parse_args(); Z = a.slice; TA = dsp.TA
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; names = [r for r in dsp.ROI_NAMES if r in rois]; z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float)
    def enh(c, t): return c - np.median(c[t < 40])
    mfc = {r: np.array([mf[..., i][rois[r]].mean() for i in range(mf.shape[-1])]) for r in names}
    tg = np.linspace(-1, 1, 2048); dt = TA / (len(tg) - 1); f = np.fft.rfftfreq(len(tg), dt); rows = []; fig, axs = plt.subplots(2, max(2, len(a.runs.split(","))), figsize=(4.2 * max(2, len(a.runs.split(","))), 7))
    for j, spec in enumerate(a.runs.split(",")):
        nm, d = spec.split(":", 1); ckp = f"{d}/model_slice_{Z:02d}.pt"; rec = f"{d}/nik_slice_{Z}_cplx.npy"; row = dict(label=nm)
        if os.path.exists(ckp):
            ck = torch.load(ckp, map_location="cpu", weights_only=False)
            args = SimpleNamespace(**{k: ck[k] for k in ("model", "rank", "hidden", "depth", "w0", "s0", "coil_embed_dim", "k_freq", "k_sigma", "t_freq", "t_sigma", "ff_seed")}, patlak_free=0,
                                   aif_file=dsp.AIF(Z), tofts_basis=dsp.BASIS(Z, 8), phi_hidden=ck.get("phi_hidden", 64), phi_depth=ck.get("phi_depth", 3), phi_w0=ck.get("phi_w0", 30.0), phi_ortho=ck.get("phi_ortho", False), n_pk=-1, radial_alpha=1.0, coil_mode=ck.get("coil_mode", "input"))
            m = build_model(args, int(ck["ncc"])); m.load_state_dict(ck["state_dict"]); m.eval()
            with torch.no_grad(): P = m.basis(torch.as_tensor(tg, dtype=torch.float32)).numpy()                          # (G, R, 2)
            at = P[..., 0].astype(np.float64) + 1j * P[..., 1].astype(np.float64); at = at - at.mean(0); pw = np.abs(np.fft.fft(at, axis=0))[: len(tg) // 2 + 1] ** 2   # complex atoms: full fft, positive half (rfft rejects complex input); hi = pw[f > 0.1].sum(0) / (pw[1:].sum(0) + 1e-12)
            row.update(rank=int(at.shape[1]), resp_frac_mean=float(hi.mean()), resp_frac_max=float(hi.max()), n_atoms_resp_gt_0p3=int((hi > 0.3).sum()))
            ax = axs[0, j]; sc = np.abs(at).max(0) + 1e-9
            for r in range(at.shape[1]): ax.plot((tg + 1) / 2 * TA, np.real(at[:, r]) / sc[r] + 1.2 * r, lw=0.6)
            ax.set_title(f"{nm}\natoms (real part, stacked), resp frac mean {hi.mean():.2f}", fontsize=8); ax.set_xlabel("t [s]"); ax.set_yticks([])
            ax = axs[1, j]; ax.bar(np.arange(len(hi)), hi, color="0.4"); ax.axhline(0.3, color="r", lw=0.6); ax.set_ylim(0, 1); ax.set_title("power above 0.1 Hz / total, per atom", fontsize=8); ax.set_xlabel("atom")
        if os.path.exists(rec):
            v = np.abs(np.load(rec)).astype(np.float32); t = (np.arange(v.shape[-1]) + 0.5) * TA / v.shape[-1]
            for r in names:
                c = np.array([v[..., i][rois[r]].mean() for i in range(v.shape[-1])]); ci = np.interp(tmf, t, c)
                row[f"{r}_osc"] = hp_std(c); row[f"{r}_nrmse"] = float(np.linalg.norm(enh(ci, tmf) - enh(mfc[r], tmf)) / (np.linalg.norm(enh(mfc[r], tmf)) + 1e-12))
            wb = f"{d}/wandb_runs/slice_{Z:02d}.json"
            if os.path.exists(wb):
                try: w = json.load(open(wb)); row["best_heldout"] = float(w.get("summary", w).get("best_heldout_mse", np.nan))
                except Exception: pass
        rows.append(row); print(row, flush=True)
    for r in names: rows.insert(0, {"label": f"model-free ({r} osc only)", f"{r}_osc": hp_std(mfc[r])}) if r == names[0] else None
    K = ["rank", "resp_frac_mean", "resp_frac_max", "n_atoms_resp_gt_0p3"] + [f"{r}_osc" for r in names] + [f"{r}_nrmse" for r in names] + ["best_heldout"]
    L = [f"# sub16 atom diagnostic, {dsp.DS} slice {Z}: respiratory content of the learned atoms (power above 0.1 Hz / total) and oscillation of the roi curves (high-pass temporal std / mean) vs curve nrmse", "", "| run | " + " | ".join(K) + " |", "|---|" + "---|" * len(K)]
    for r in rows: L.append(f"| {r['label']} | " + " | ".join((f"{r[k]:.3f}" if isinstance(r.get(k), float) else str(r[k])) if k in r else "-" for k in K) + " |")
    out = f"{B}/results/tofts_vs_patlak/sub16_atoms{dsp.SFX}_sl{Z}{a.tag}"; open(out + ".md", "w").write("\n".join(L) + "\n"); json.dump(rows, open(out + ".json", "w"), indent=1)
    fig.suptitle(f"{dsp.DS} slice {Z}: learned atoms and their respiratory content per run", fontsize=10); fig.tight_layout(); fig.savefig(out + ".png", dpi=120, facecolor="white"); print("\n".join(L)); print("ATOMS_DONE")

if __name__ == "__main__": main()
