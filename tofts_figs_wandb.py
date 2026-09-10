"""in vivo figures for the tofts arms (patlak, tofts, tofts8, whatever is in results/tofts_vs_patlak/invivo) -> wandb,
project dce_nik, group tofts_figs, one run per slice: panel (3 phases x arms, seed 0, + curves), roi overlay on the
anatomy, all-seed roi curves vs model-free, peak-frame difference vs model-free, metric bars and test annuli from
invivo<suffix>.json. offline fallback: pngs in results/tofts_vs_patlak/figures_wandb/.
usage: python tofts_figs_wandb.py --slices 18,19,21 [--suffix _r8] [--tag <label>]"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, glob, argparse, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; RES = f"{B}/results/tofts_vs_patlak"; IV = f"{RES}/invivo"
GV = "/net/beegfs/users/P101440/grasp_v2/results_grasp_v2"; GP = "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs"
TA = 375.0; ROIS = ("aorta", "cortex", "medulla", "liver")
REFSETS = {"f25": (("grasp v2 f25 (488 sp, 122 fr, lam0.25)", "{GV}/gv2_slice{Z}_f25.npy"), ("grasp pro f25 (488 sp, 122 fr, K5)", "{GP}/cs_slice{Z}_f25.npy")),
           "k80": (("grasp v2 k80 (1368 views, 142 fr, n12 lam0.25)", "{GV}/gv2_slice{Z}_n12_k80.npy"), ("grasp pro f80match (1368 views, 122 fr, K5)", "{GP}/cs_slice{Z}_f80match.npy"))}
REFSET = "f25"; SPK_LABEL = "keep_f25 (488 of 1710 spokes)"                  # set by main (--refs)
COL = {"model-free": "k", "patlak": "#0369a1", "tofts8": "#1e8449", "tofts": "#c0392b", "grasp v2": "#e67e22", "grasp pro": "#8e44ad"}
PHASES = [("pre ~20s", 20.0), ("peak ~65s", 65.0), ("late ~250s", 250.0)]
KEYS = ["mf_aorta_affine", "mf_cortex_affine", "mf_medulla_affine", "mf_liver_affine", "aorta_peak_ratio_vs_mf", "cortex_medulla_late_corr", "val_kNMSE", "test_kNMSE"]

def ft(nt): e = np.linspace(0, TA, nt+1); return 0.5*(e[:-1]+e[1:])
def bs(c): return c - np.median(c[:8])
def col(nm):
    nm = str(nm).lower().replace("-", " ")
    for k in COL:
        if nm.startswith(k): return COL[k]
    return "0.5"

class Slice:
    """model-free reference, rois, body, anatomy; loaders and the one global scale"""
    def __init__(self, Z):
        sys.path.insert(0, B); import consolidated as C
        self.Z = Z; ctx = C.slice_ctx(Z); md = np.load(f"{B}/step2_slice{Z}.npz"); self.mf = md["mf"]; self.tmf = md["tmf"]
        self.rois = dict(ctx["rois"]); self.body = ctx["BODY"]; a = np.asarray(ctx["cs100"]); self.anat = a.mean(-1) if a.ndim == 3 else a
        self.mfv = np.transpose(self.mf, (1, 2, 0)).astype(np.float32)
    def refwin(self, nt):
        e = np.linspace(0, TA, nt+1); o = np.zeros((self.mf.shape[1], self.mf.shape[2], nt), np.float32)
        for g in range(nt):
            m = (self.tmf >= e[g]) & (self.tmf < e[g+1])
            if not m.any(): m = np.zeros_like(self.tmf, bool); m[np.argmin(np.abs(self.tmf-0.5*(e[g]+e[g+1])))] = True
            o[:, :, g] = self.mf[m].mean(0)
        return o
    def sc(self, v): R = self.refwin(v.shape[-1]); b = self.body; return v * (np.sum(v[b]*R[b]) / (np.sum(v[b]**2) + 1e-12))
    def curve(self, v, t, roi): c = np.array([v[..., i][self.rois[roi]].mean() for i in range(v.shape[-1])]); return bs(np.interp(self.tmf, t, c))
    def load(self):
        """arms {arm: {seed: (vol, t)}} in order patlak, tofts, tofts8, rest; refs [(name, vol, t)]"""
        arms = {}
        for f in sorted(glob.glob(f"{IV}/*_sl{self.Z}_s?/nik_slice_{self.Z:02d}_cplx.npy")):
            d = os.path.basename(os.path.dirname(f)); arm = d.rsplit("_sl", 1)[0]; s = int(d[-1])
            v = self.sc(np.abs(np.load(f)).astype(np.float32)); arms.setdefault(arm, {})[s] = (v, ft(v.shape[-1]))
        order = [a for a in ("patlak", "tofts", "tofts8") if a in arms] + sorted(a for a in arms if a not in ("patlak", "tofts", "tofts8"))
        refs = []
        for nm, pt in REFSETS[REFSET]:
            p = pt.format(GV=GV, GP=GP, Z=self.Z)
            if os.path.exists(p): v = self.sc(np.abs(np.load(p)).astype(np.float32)); refs.append((nm, v, ft(v.shape[-1])))
        return {a: arms[a] for a in order}, refs

def fig_panel(S, arms, refs):
    M = [("model-free nufft (31-spoke window)", S.mfv, S.tmf)] + [(f"{a} nik s0", arms[a][0][0], arms[a][0][1]) for a in arms if 0 in arms[a]] + refs
    n = len(M); nc = max(n, 3); fig = plt.figure(figsize=(3.0*nc, 10.5)); gs = fig.add_gridspec(4, nc, height_ratios=[1, 1, 1, 1.35], hspace=0.16, wspace=0.04)
    vmax = float(np.percentile(S.mfv[S.body], 99.5))
    for pi, (plab, pt) in enumerate(PHASES):
        for mi, (nm, v, t) in enumerate(M):
            ax = fig.add_subplot(gs[pi, mi]); i = int(np.argmin(np.abs(np.asarray(t) - pt))); ax.imshow(v[:, :, i], cmap="gray", vmin=0, vmax=vmax); ax.axis("off")
            if pi == 0: ax.set_title(nm, fontsize=8)
            if mi == 0: ax.text(-0.08, 0.5, plab, transform=ax.transAxes, rotation=90, va="center", fontsize=9)
    for ri, r in enumerate(("aorta", "cortex", "medulla")):
        ax = fig.add_subplot(gs[3, ri*nc//3:(ri+1)*nc//3])
        for mi, (nm, v, t) in enumerate(M): ax.plot(S.tmf, S.curve(v, t, r), color=col(nm), lw=2.2 if mi == 0 else 1.4, ls="-" if mi == 0 else "--", label=nm)
        ax.set_title(r, fontsize=9); ax.set_xlabel("time (s)", fontsize=8); ax.grid(alpha=.3); ax.tick_params(labelsize=7)
        if ri == 0: ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(f"in vivo slice {S.Z}: 3 phases, seed 0, one global scale vs model-free, {SPK_LABEL}; curves baseline subtracted", fontsize=9); return fig

def fig_rois(S):
    fig, ax = plt.subplots(figsize=(5.2, 5.2)); a = S.anat; ax.imshow(a, cmap="gray", vmin=0, vmax=float(np.percentile(a[S.body], 99.5))); ax.axis("off")
    for r, c in zip(ROIS, ("#e74c3c", "#3498db", "#2ecc71", "#f1c40f")):
        if r in S.rois and S.rois[r].any(): ax.contour(S.rois[r].astype(float), levels=[0.5], colors=[c], linewidths=1.2); ax.plot([], [], color=c, label=f"{r} ({int(S.rois[r].sum())} px)")
    ax.contour(S.body.astype(float), levels=[0.5], colors=["w"], linewidths=0.5, linestyles="dotted"); ax.legend(fontsize=7, loc="lower right")
    ax.set_title(f"slice {S.Z}: rois on the grasp-pro 100% image (same masks for every arm)", fontsize=9); return fig

def fig_curves(S, arms, refs):
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, r in zip(axs, ROIS):
        if r not in S.rois or not S.rois[r].any(): ax.axis("off"); continue
        ax.plot(S.tmf, bs(np.array([im[S.rois[r]].mean() for im in S.mf])), "k", lw=2.4, label="model-free")
        for nm, v, t in refs: ax.plot(S.tmf, S.curve(v, t, r), color=col(nm), lw=1.4, ls="--", label=nm.split(" (")[0])
        for a in arms:
            for s, (v, t) in sorted(arms[a].items()): ax.plot(S.tmf, S.curve(v, t, r), color=COL.get(a, "0.5"), lw=1.0, alpha=0.85, label=f"{a} nik ({len(arms[a])} seeds)" if s == min(arms[a]) else None)
        ax.set_title(r, fontsize=10); ax.set_xlabel("time (s)"); ax.grid(alpha=.3)
    axs[0].set_ylabel("enhancement, baseline subtracted"); axs[0].legend(fontsize=7)
    fig.suptitle(f"slice {S.Z}: roi curves, all seeds, raw scale (one global ls scale vs model-free, no per-roi normalization)", fontsize=10); fig.tight_layout(); return fig

def fig_peakdiff(S, arms, refs, pt=65.0):
    M = [(f"{a} nik s0", arms[a][0][0], arms[a][0][1]) for a in arms if 0 in arms[a]] + refs
    fig, axs = plt.subplots(2, len(M), figsize=(3.2*len(M), 6.6), squeeze=False)
    vmax = float(np.percentile(S.mfv[S.body], 99.5)); i0 = int(np.argmin(np.abs(S.tmf - pt))); ref = S.mf[max(0, i0-3):i0+4].mean(0)
    for j, (nm, v, t) in enumerate(M):
        i = int(np.argmin(np.abs(np.asarray(t) - pt))); im = v[:, :, i]; d = im - ref
        axs[0, j].imshow(im, cmap="gray", vmin=0, vmax=vmax); axs[0, j].set_title(nm, fontsize=8); axs[0, j].axis("off")
        axs[1, j].imshow(d, cmap="RdBu_r", vmin=-0.4*vmax, vmax=0.4*vmax); axs[1, j].axis("off")
        axs[1, j].set_title(f"minus model-free, body nrmse {np.linalg.norm(d[S.body])/np.linalg.norm(ref[S.body]):.3f}", fontsize=8)
    fig.suptitle(f"slice {S.Z}: frame nearest {pt:.0f}s vs model-free (7-frame mean around it); model-free streaks are part of the reference", fontsize=9); return fig

def fig_metrics(Z, rows):
    R = [r for r in rows if r.get("slice") == Z and r.get("status") == "complete"]; arms = list(dict.fromkeys(r["arm"] for r in R if r["seed"] >= 0))
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.6)); axs = axs.ravel()
    for ax, k in zip(axs, KEYS):
        for j, a in enumerate(arms):
            v = np.array([r[k] for r in R if r["arm"] == a and r["seed"] >= 0 and k in r], float)
            if v.size: ax.bar(j, np.nanmean(v), yerr=np.nanstd(v), color=COL.get(a, "0.5"), alpha=0.85, capsize=3); ax.scatter([j]*v.size, v, s=10, color="k", zorder=3)
        for r in R:
            if r["seed"] < 0 and k in r and r[k] is not None and np.isfinite(r[k]): ax.axhline(r[k], color=col(r["arm"]), ls="--", lw=1, label=str(r["arm"]).split(" (")[0])
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, fontsize=8); ax.set_title(k, fontsize=9); ax.grid(alpha=.3, axis="y")
    axs[0].legend(fontsize=6); fig.suptitle(f"slice {Z}: bars mean ± sd over seeds, dots seeds, dashed grasp refs. mf_* = nrmse vs model-free (lower better); kNMSE = held-out spokes (nik only)", fontsize=9); fig.tight_layout(); return fig

def fig_annuli(Z, rows):
    E = np.linspace(0, 1, 17); fig, ax = plt.subplots(figsize=(7, 3.8)); arms = list(dict.fromkeys(r["arm"] for r in rows if r.get("slice") == Z and "test_annuli" in r))
    if not arms: plt.close(fig); return None
    for a in arms:
        A = np.array([r["test_annuli"] for r in rows if r.get("slice") == Z and r["arm"] == a]); ax.plot(0.5*(E[:-1]+E[1:]), A.mean(0), "-o", ms=3, color=COL.get(a, "0.5"), label=f"{a} ({A.shape[0]} seeds)")
    ax.set_xlabel("|k| / kmax (annulus centre)"); ax.set_ylabel("test-spoke k-space nmse"); ax.set_yscale("log"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"slice {Z}: held-out test spokes, nmse per |k| annulus (energy is k-centre dominated)", fontsize=9); fig.tight_layout(); return fig

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--slices", default="18,19,21"); ap.add_argument("--suffix", default="_r8", help="invivo<suffix>.json, falls back to invivo.json")
    ap.add_argument("--tag", default=os.environ.get("SLURM_JOB_ID", "local")); ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--iv", default="invivo", help="run dir under results/tofts_vs_patlak: invivo (f25) or invivo_k80"); ap.add_argument("--refs", default="f25", choices=list(REFSETS)); a = ap.parse_args()
    global IV, REFSET, SPK_LABEL; IV = f"{RES}/{a.iv}"; REFSET = a.refs; spk = "k80 (1368 of 1708 views, v%10<8)" if a.refs == "k80" else "keep_f25 (488 of 1710 spokes)"; SPK_LABEL = spk
    import nik_wandb as W
    jf = f"{RES}/invivo{a.suffix}.json"; jf = jf if os.path.exists(jf) else f"{RES}/invivo.json"; rows = json.load(open(jf)) if os.path.exists(jf) else []
    print("metrics from", jf, len(rows), "rows", flush=True); FB = f"{RES}/figures_wandb"
    for Z in [int(z) for z in a.slices.split(",")]:
        S = Slice(Z); arms, refs = S.load(); print(f"slice {Z}: arms {[(k, sorted(v)) for k, v in arms.items()]}, refs {[r[0] for r in refs]}", flush=True)
        run = W.Run(f"figs_{a.iv}_sl{Z}_{a.tag}", config=dict(slice=Z, arms=list(arms), seeds={k: sorted(v) for k, v in arms.items()}, refs=[r[0] for r in refs], metrics_json=jf, spokes=spk),
                    group="tofts_figs", tags=["invivo", f"sl{Z}", a.refs], local_json=f"{FB}/figs_{a.iv}_sl{Z}_{a.tag}.json", enabled=not a.no_wandb)
        figs = {"panel": fig_panel(S, arms, refs), "roi_overlay": fig_rois(S), "curves_all_seeds": fig_curves(S, arms, refs), "peak_diff": fig_peakdiff(S, arms, refs)}
        if rows:
            figs["metrics"] = fig_metrics(Z, rows); an = fig_annuli(Z, rows)
            if an is not None: figs["annuli"] = an
        if run.wb is not None:
            import wandb
            run.wb.log({k: wandb.Image(f) for k, f in figs.items()})
            if rows:
                cols = ["slice", "arm", "seed"] + KEYS + ["wall_s", "peak_gpu_mb", "params"]
                run.wb.log({"metrics_table": wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows if r.get("slice") == Z])})
            print(f"slice {Z}: {len(figs)} figures -> {getattr(run.wb, 'url', 'wandb')}", flush=True)
        else:
            os.makedirs(FB, exist_ok=True)
            for k, f in figs.items(): f.savefig(f"{FB}/sl{Z}_{k}_{a.tag}.png", dpi=110, bbox_inches="tight")
            print(f"slice {Z}: wandb unavailable, pngs in {FB}", flush=True)
        for f in figs.values(): plt.close(f)
        run.finish(n_figs=len(figs))
    print("FIGS_DONE")

if __name__ == "__main__": main()
