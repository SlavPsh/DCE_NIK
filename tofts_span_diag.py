"""in vivo amplitude deficit of the pk-basis arms: is the tofts basis span the limit, or the training?
projects the model-free (31-spoke nufft) roi curves and every body voxel curve onto the fixed tofts atoms at ranks 3/5/8/12 (least squares,
same atoms the network uses, interpolated at the frame times) and compares the retained first-pass and washout amplitude with what the
trained nik-tofts / nik-tofts8 / nik-patlak arms produce on the same k80 input. also checks the basis aif against the model-free aorta curve.
out: results/tofts_vs_patlak/span_diag_sl<Z>.{md,json}, figures/span_diag_sl<Z>.png
usage: python tofts_span_diag.py --slice 21"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, B)
import dsp                                                                                   # dataset paths (DCE_DS=p3 default / p8)
import consolidated as C
from story_figs import ls_scale

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slice", type=int, default=21); ap.add_argument("--arms", default="tofts,tofts8,patlak"); a = ap.parse_args(); Z = a.slice
    ctx = C.slice_ctx(Z); rois = ctx["rois"]; body = ctx["BODY"]; z = np.load(dsp.STEP2(Z)); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"], float); TA = dsp.TA
    names = [r for r in dsp.ROI_NAMES + (dsp.STATIC,) if r in rois]
    bz = np.load(dsp.BASIS(Z, 8) if dsp.DS != "p3" else f"{B}/results/tofts_vs_patlak/basis_sl{Z}.npz", allow_pickle=True); atoms, tg = np.asarray(bz["atoms"], float), np.asarray(bz["tgrid_s"], float)
    Phi = np.stack([np.interp(tmf, tg, atoms[:, k]) for k in range(atoms.shape[1])], 1)                       # [F,R] atoms at the model-free frame times
    pre = tmf < 40; fp = (tmf > 40) & (tmf < 110); wo = tmf > 200
    def roi_curve(v, t): return {r: np.array([v[..., i][rois[r]].mean() for i in range(v.shape[-1])]) for r in names}
    def enh(c, t): return c - np.median(c[t < 40])
    def amp(c, t): e = enh(c, t); return dict(peak=float(e[(t > 20) & (t < 210)].max()), washout=float(e[t > 200].mean()))
    mfc = roi_curve(mf, tmf); ref = {r: amp(mfc[r], tmf) for r in names}
    out = dict(slice=Z, ranks={}, arms={}, aif={}); curves = {r: {"model-free": (tmf, enh(mfc[r], tmf))} for r in names}
    # projection of the model-free curves onto the span at each rank (Q orthonormal at tmf; least squares = projection)
    X = mf.reshape(-1, mf.shape[-1])[body.ravel()]
    for Rk in (3, 5, 8, 12):
        if Rk > Phi.shape[1]: continue
        Q, _ = np.linalg.qr(Phi[:, :Rk]); P = lambda c: (c @ Q) @ Q.T
        Xh = P(X); err = float(np.sqrt(((X - Xh) ** 2).sum() / (X ** 2).sum())); errd = float(np.sqrt((((X - Xh)[:, ~pre]) ** 2).sum() / (((X - X[:, pre].mean(1, keepdims=True))[:, ~pre]) ** 2).sum()))
        row = dict(err_all=err, err_dynamic=errd)
        for r in names:
            ph = P(mfc[r]); ar = amp(ph, tmf); row[r] = dict(peak_ratio=ar["peak"] / (ref[r]["peak"] + 1e-9), washout_ratio=ar["washout"] / (ref[r]["washout"] + 1e-9), nrmse=float(np.linalg.norm(ph - mfc[r]) / np.linalg.norm(enh(mfc[r], tmf))))
            curves[r][f"proj r{Rk}"] = (tmf, enh(ph, tmf))
        out["ranks"][Rk] = row
    # what the trained arms actually produce (same k80 input), one global scale vs the windowed model-free reference, roi enhancement
    def ft(nt): e = np.linspace(0, TA, nt + 1); return 0.5 * (e[:-1] + e[1:])
    def refwin(nt):
        e = np.linspace(0, TA, nt + 1); o = np.zeros(mf.shape[:2] + (nt,), np.float32)
        for g in range(nt):
            m = (tmf >= e[g]) & (tmf < e[g + 1])
            if not m.any(): m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf - 0.5 * (e[g] + e[g + 1])))] = True
            o[:, :, g] = mf[:, :, m].mean(2)
        return o
    for arm in a.arms.split(","):
        p = f"{B}/results/tofts_vs_patlak/{os.environ.get('IV_DIR', 'invivo_k80')}/{arm}_sl{Z}_s0/nik_slice_{Z}_cplx.npy"
        if not os.path.exists(p): print("missing", p); continue
        v = np.abs(np.load(p)).astype(np.float32); t = ft(v.shape[-1]); v = ls_scale(v, refwin(v.shape[-1]), body); c = roi_curve(v, t); row = {}
        for r in names:
            ar = amp(c[r], t); row[r] = dict(peak_ratio=ar["peak"] / (ref[r]["peak"] + 1e-9), washout_ratio=ar["washout"] / (ref[r]["washout"] + 1e-9)); curves[r][f"NIK-{arm}"] = (t, enh(c[r], t))
        out["arms"][arm] = row
    # basis aif vs the model-free aorta curve: timing and plateau
    az = np.load(dsp.AIF(Z), allow_pickle=True); af, tC = np.asarray(az["aif_frame"], float), np.asarray(az["tC"], float); ao = enh(mfc["aorta"], tmf)
    afe = af - np.median(af[tC < 40]); out["aif"] = dict(ttp_basis_s=float(tC[np.argmax(afe)]), ttp_mf_s=float(tmf[np.argmax(ao)]), plateau_over_peak_basis=float(afe[tC > 200].mean() / afe.max()), plateau_over_peak_mf=float(ao[tmf > 200].mean() / ao.max()))
    curves["aorta"]["basis aif (scaled)"] = (tC, afe / afe.max() * ao.max())
    # report
    L = [f"# tofts basis span vs trained arms, in vivo slice {Z}, k80 (reference = model-free 31-spoke nufft, roi enhancement, baseline subtracted)", "",
         "ratios are amplitude relative to the model-free curve: first-pass peak (20 to 210 s) and washout mean (t > 200 s). projection = least squares of the model-free curve onto the first R atoms at the frame times (what a perfect fit inside the span could reach).", "",
         "| curve | " + " | ".join(f"{r} peak / washout" for r in names) + " | body voxel err (all / dynamic part) |", "|---|" + "---|" * (len(names) + 1)]
    for Rk, row in out["ranks"].items(): L.append(f"| projection rank {Rk} | " + " | ".join(f"{row[r]['peak_ratio']:.2f} / {row[r]['washout_ratio']:.2f}" for r in names) + f" | {row['err_all']:.3f} / {row['err_dynamic']:.3f} |")
    for arm, row in out["arms"].items(): L.append(f"| NIK-{arm} (trained) | " + " | ".join(f"{row[r]['peak_ratio']:.2f} / {row[r]['washout_ratio']:.2f}" for r in names) + " | |")
    ai = out["aif"]; L += ["", f"basis aif vs model-free aorta: time to peak {ai['ttp_basis_s']:.1f} s vs {ai['ttp_mf_s']:.1f} s; plateau / peak {ai['plateau_over_peak_basis']:.2f} vs {ai['plateau_over_peak_mf']:.2f}"]
    R = f"{B}/results/tofts_vs_patlak"; SD = os.environ.get("STORY_TAG", ""); open(f"{R}/span_diag_sl{Z}{SD}.md", "w").write("\n".join(L)); json.dump(out, open(f"{R}/span_diag_sl{Z}{SD}.json", "w"), indent=1); print("\n".join(L))
    fig, ax = plt.subplots(1, len(names), figsize=(5.2 * len(names), 4.2))
    for j, r in enumerate(names):
        for k, (nm, (t, c)) in enumerate(curves[r].items()):
            ax[j].plot(t, c, lw=2.4 if nm == "model-free" else 1.3, color="k" if nm == "model-free" else None, ls="--" if nm.startswith("proj") else ("-" if nm.startswith("NIK") else ":"), label=nm)
        ax[j].set_title(r); ax[j].set_xlabel("t [s]"); ax[j].axhline(0, color="0.7", lw=0.6)
        if j == 0: ax[j].set_ylabel("enhancement"); ax[j].legend(fontsize=7)
    fig.suptitle(f"slice {Z}, k80: model-free curves projected onto the tofts atoms (dashed) vs the trained arms (solid)", fontsize=11); fig.tight_layout()
    os.makedirs(f"{R}/figures", exist_ok=True); fig.savefig(f"{R}/figures/span_diag_sl{Z}{SD}.png", dpi=140, facecolor="white"); print("SPAN_DIAG_DONE")

if __name__ == "__main__": main()
