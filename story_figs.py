"""presentation story figures, identical input per domain.
phantom (xcat no-motion, truth): one figure per nik family (free, sub16, patlak, tofts): truth | nik | grasp pro k12 | grasp 25 spf
at one time point, body-masked haarpsi / psnr / ssim vs truth (68-frame window, one global ls scale, as in the report tables),
aorta / cortex / medulla curves. every method on the same 5 of 7 spokes per frame.
in vivo (slice 21, k80 = 1368 of 1708 views for every method): model-free | 4 nik | grasp pro | grasp at one time point, curves vs model-free.
usage: XPH_SIM=nomotion python story_figs.py [--t-phantom 90] [--t-invivo 90] [--tofts tofts|tofts8]"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, argparse, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/net/beegfs/users/P101440/DCE_NIK")
from masked_metrics import haarpsi_masked, ssim_masked
B = "/net/beegfs/users/P101440/DCE_NIK"; GV = "/net/beegfs/users/P101440/grasp_v2/results_grasp_v2"; GP = "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COL = {"NIK-free": "#00838f", "NIK-sub16": "#1e8449", "NIK-patlak": "#0369a1", "NIK-tofts": "#c0392b", "NIK-tofts8": "#7b1fa2", "GRASP-Pro": "#8e44ad", "GRASP": "#e67e22"}
ROIS = ("aorta", "cortex", "medulla")

def winavg(v, G):
    n = v.shape[-1] // G; return np.stack([v[:, :, g*G:(g+1)*G].mean(2) for g in range(n)], -1)

def ls_scale(v, ref, body):
    return v * (np.sum(v[body] * ref[body]) / (np.sum(v[body]**2) + 1e-12))

def metrics(img, ref, body):
    """body-masked haarpsi / ssim over 40 frames, psnr over all frames; img and ref on the same frame grid"""
    mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev); hs, ss = [], []; nt = img.shape[-1]
    for t in range(0, nt, max(1, nt // 40)):
        vmax = float(np.percentile(ref[:, :, t][body], 99.5)) + 1e-12
        a = torch.from_numpy(np.clip(img[:, :, t] / vmax, 0, 1)[None, None]).float().to(dev); b = torch.from_numpy(np.clip(ref[:, :, t] / vmax, 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(a, b, mt, data_range=1.0).cpu())); ss.append(float(ssim_masked(a, b, mt, data_range=1.0).cpu()))
    pk = float(ref[body].max()); mse = float(np.mean([((img[:, :, t][body] - ref[:, :, t][body])**2).mean() for t in range(nt)]))
    return dict(haarpsi=float(np.mean(hs)), ssim=float(np.mean(ss)), psnr=10 * np.log10(pk**2 / (mse + 1e-20)))

def panel(fig_path, title, M, rois, body, t_show, tref, ref_curves, ylab, note, metric_rows=None):
    """M: list of (name, vol[H,W,T], t[T]); first entry is the reference (truth or model-free)"""
    n = len(M); fig = plt.figure(figsize=(3.3 * n, 8.6))
    gt = fig.add_gridspec(1, n, top=0.9, bottom=0.53, wspace=0.05); gb = fig.add_gridspec(1, 3, top=0.42, bottom=0.09, wspace=0.28, left=0.06, right=0.98)
    vmax = float(np.percentile(M[0][1][:, :, int(np.argmin(np.abs(M[0][2] - t_show)))][body], 99.5))
    for j, (nm, v, t) in enumerate(M):
        ax = fig.add_subplot(gt[0, j]); i = int(np.argmin(np.abs(np.asarray(t) - t_show))); ax.imshow(v[:, :, i], cmap="gray", vmin=0, vmax=vmax); ax.axis("off")
        ax.set_title(nm, fontsize=13, fontweight="bold", color=COL.get(nm, "k"))
        if metric_rows and nm in metric_rows:
            m = metric_rows[nm]; ax.text(0.5, -0.03, f"HaarPSI {m['haarpsi']:.3f}   SSIM {m['ssim']:.3f}\nPSNR {m['psnr']:.1f} dB", transform=ax.transAxes, ha="center", va="top", fontsize=10, linespacing=1.4)
    for r, roi in enumerate(ROIS):
        ax = fig.add_subplot(gb[0, r]); ax.plot(tref, ref_curves[roi], "k", lw=2.6, label=M[0][0])
        for nm, v, t in M[1:]:
            c = np.array([v[..., i][rois[roi]].mean() for i in range(v.shape[-1])]); c = np.interp(tref, t, c)
            if ylab.startswith("enhancement"): c = c - np.median(c[:8])
            ax.plot(tref, c, color=COL.get(nm, "0.5"), lw=1.8, ls="-" if nm.startswith("NIK") else "--", label=nm)
        ax.set_title(roi, fontsize=12); ax.set_xlabel("time (s)", fontsize=11); ax.grid(alpha=.3); ax.tick_params(labelsize=9)
        if r == 0: ax.set_ylabel(ylab, fontsize=11); ax.legend(fontsize=9, loc="upper right")
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.97); fig.text(0.5, 0.015, note, ha="center", fontsize=10, color="#546e7a")
    fig.savefig(fig_path, dpi=150, facecolor="white"); plt.close(fig); print("saved", fig_path, flush=True)

def phantom(t_show):
    os.environ.setdefault("XPH_SIM", "nomotion"); import xph_pipeline as P, xph_common as X
    d = P.data(); tq = np.asarray(d["times"]); body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq).astype(np.float32); Rz = X.rois(P.ZI, d["labels"]); A = f"{P.OUT}/arrays"; G = 5
    Tw = winavg(Tr, G); tw = np.array([tq[g*G:(g+1)*G].mean() for g in range(Tw.shape[-1])])
    truth_c = {r: Tr[Rz[r]].mean(0) for r in ROIS}
    src = {"NIK-free": (f"{A}/nik_fine_free.npy", None), "NIK-sub16": (f"{A}/nik_fine_sub16.npy", None),
           "NIK-patlak": (f"{A}/nik_eval_w768_ks2.5_s0.npz", "rec_best"), "NIK-tofts": (f"{A}/nik_eval_w768_ks2.5_s0_tofts16.npz", "rec_best")}
    def load(p, key): z = np.load(p, allow_pickle=True); return np.abs(z[key] if key else z).astype(np.float32)
    gp = ls_scale(load(f"{A}/grasp_recon.npz", "rec"), Tr, body); gv = ls_scale(np.abs(np.load(f"{P.OUT}/v2_sweep/v2_G05.npy")).astype(np.float32), Tw, body)
    mrows = {"GRASP-Pro": metrics(winavg(gp, G), Tw, body), "GRASP": metrics(gv, Tw, body)}
    for nm, (p, key) in src.items():
        v = ls_scale(load(p, key), Tr, body); mrows[nm] = metrics(winavg(v, G), Tw, body)
        M = [("truth", Tr, tq), (nm, v, tq), ("GRASP-Pro", gp, tq), ("GRASP", gv, tw)]
        panel(f"{P.OUT}/figures/story_phantom_{nm.split('-')[1]}.png", f"phantom, {nm} vs GRASP-Pro and GRASP, vs ground truth",
              M, Rz, body, t_show, tq, truth_c, "signal", f"same input for every method: 5 of 7 spokes per frame (71%); image at t = {t_show:.0f} s; metrics on the 68-frame window, body-masked, one global scale", mrows)
    print("phantom metrics:", {k: {kk: round(vv, 3) for kk, vv in m.items()} for k, m in mrows.items()})

def invivo(t_show, tofts_arm):
    import consolidated as C
    ctx = C.slice_ctx(21); rois = ctx["rois"]; body = ctx["BODY"]; z = np.load(f"{B}/step2_slice21.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32); tmf = np.asarray(z["tmf"]).astype(np.float64); TA = 375.0
    def ft(nt): e = np.linspace(0, TA, nt + 1); return 0.5 * (e[:-1] + e[1:])
    def refwin(nt):
        e = np.linspace(0, TA, nt + 1); out = np.zeros((mf.shape[0], mf.shape[1], nt), np.float32)
        for g in range(nt):
            m = (tmf >= e[g]) & (tmf < e[g+1])
            if not m.any(): m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf - 0.5*(e[g]+e[g+1])))] = True
            out[:, :, g] = mf[:, :, m].mean(2)
        return out
    def sc(v): return ls_scale(v, refwin(v.shape[-1]), body)
    L = lambda p: sc(np.abs(np.load(p)).astype(np.float32))
    IV = f"{B}/results/tofts_vs_patlak/invivo_k80"
    M = [("model-free", mf, tmf),
         ("NIK-free", L(f"{B}/results_sl21_k80/nik_slice_21.npy"), None), ("NIK-sub16", L(f"{B}/results/realdata_nik_vs_cs_figures/outcoil_subspace_output_slice21.npy"), None),
         ("NIK-patlak", L(f"{IV}/patlak_sl21_s0/nik_slice_21_cplx.npy"), None), ("NIK-" + tofts_arm, L(f"{IV}/{tofts_arm}_sl21_s0/nik_slice_21_cplx.npy"), None),
         ("GRASP-Pro", L(f"{GP}/cs_slice21_f80match.npy"), None), ("GRASP", L(f"{GV}/gv2_slice21_n12_k80.npy"), None)]
    M = [(nm, v, (t if t is not None else ft(v.shape[-1]))) for nm, v, t in M]
    mfc = {r: (lambda c: c - np.median(c[:8]))(np.array([mf[..., i][rois[r]].mean() for i in range(mf.shape[-1])])) for r in ROIS}
    panel(f"{B}/results/realdata_nik_vs_cs_figures/figures/story_invivo_k80_sl21.png", "in vivo slice 21: four NIK families, GRASP-Pro and GRASP on the same k80 input",
          M, rois, body, t_show, tmf, mfc, "enhancement (baseline subtracted)", f"same input for every method: k80 = 1368 of 1708 views (v%10<8); reference = model-free nufft (31-spoke window); one global scale vs the reference; image at t = {t_show:.0f} s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--t-phantom", type=float, default=90.0); ap.add_argument("--t-invivo", type=float, default=90.0); ap.add_argument("--tofts", default="tofts"); ap.add_argument("--only", default="both")
    a = ap.parse_args()
    if a.only in ("both", "phantom"): phantom(a.t_phantom)
    if a.only in ("both", "invivo"): invivo(a.t_invivo, a.tofts)
    print("STORY_DONE")
