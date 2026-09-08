"""figures for notebook section 10 (tofts vs patlak). phantom: truth | patlak | tofts | grasp v2, 3 phases, 68-frame window + curves.
in vivo: model-free | patlak | tofts | grasp v2 f25 + curves vs model-free. seed 0 shown, tables carry all seeds."""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/scratch/rnga/vvpshenov/DCE_NIK"; RES = f"{B}/results/tofts_vs_patlak"
COL = {"truth": "k", "model-free": "k", "patlak": "#0369a1", "tofts": "#c0392b", "grasp": "#e67e22"}

def panel(M, phases, rois, tgrid_ref, fig_path, title, curve_t=None, ylab="enhancement"):
    """M = list of (name, vol[H,W,T], t[T]); rois dict; curves on tgrid_ref by interpolation."""
    n = len(M); fig = plt.figure(figsize=(3.1*n, 10.6)); gs = fig.add_gridspec(4, n, height_ratios=[1, 1, 1, 1.35], hspace=0.16, wspace=0.04)
    ref = M[0][1]; body = rois["_body"]; vmax = float(np.percentile(ref[body], 99.5))
    for pi, (plab, pt) in enumerate(phases):
        for mi, (nm, v, t) in enumerate(M):
            ax = fig.add_subplot(gs[pi, mi]); i = int(np.argmin(np.abs(np.asarray(t)-pt)))
            ax.imshow(v[:, :, i], cmap="gray", vmin=0, vmax=vmax); ax.axis("off")
            if pi == 0: ax.set_title(nm, fontsize=8)
            if mi == 0: ax.text(-0.08, 0.5, plab, transform=ax.transAxes, rotation=90, va="center", fontsize=9)
    names = [r for r in ("aorta", "cortex", "medulla") if r in rois]
    for ri, r in enumerate(names):
        ax = fig.add_subplot(gs[3, ri*n//3:(ri+1)*n//3])
        for mi, (nm, v, t) in enumerate(M):
            c = np.array([v[..., i][rois[r]].mean() for i in range(v.shape[-1])]); c = np.interp(tgrid_ref, t, c)
            if curve_t is not None: c = curve_t(c)
            key = nm.split()[0]; ax.plot(tgrid_ref, c, color=COL.get(key, "0.5"), lw=2.2 if mi == 0 else 1.5, ls="-" if mi == 0 else "--", label=nm)
        ax.set_title(f"{r} {ylab}", fontsize=9); ax.set_xlabel("time (s)", fontsize=8); ax.grid(alpha=.3); ax.tick_params(labelsize=7)
        if ri == 0: ax.legend(fontsize=6, loc="upper right")
    fig.suptitle(title, fontsize=9); fig.savefig(fig_path, dpi=110, bbox_inches="tight"); plt.close(fig); print("saved", fig_path, flush=True)

def phantom(sim):
    os.environ["XPH_SIM"] = sim
    for m in list(sys.modules):
        if m.startswith("xph_"): del sys.modules[m]
    import xph_pipeline as P, xph_common as X
    d = P.data(); tq = d["times"]; body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"]); G = 5
    wa = lambda v: np.stack([v[:, :, g*G:(g+1)*G].mean(2) for g in range(v.shape[-1]//G)], -1)
    tw = np.array([tq[g*G:(g+1)*G].mean() for g in range(len(tq)//G)]); Tw = wa(Tr)
    sc = lambda v: v * (np.sum(v[body]*Tw[body]) / (np.sum(v[body]**2) + 1e-12))
    M = [("truth (68 fr window)", Tw, tw)]
    for arm, tag in (("patlak", "w768_ks2.5_s0"), ("tofts", "w768_ks2.5_s0_tofts16")):
        ev = np.load(f"{P.OUT}/arrays/nik_eval_{tag}.npz", allow_pickle=True); M.append((f"{arm} nik s0 (rank {3 if arm=='patlak' else 16}, 5 angles/fr)", sc(wa(ev["rec_best"].astype(np.float32))), tw))
    cs = f"{P.OUT}/v2_sweep/v2_G05.npy"
    if os.path.exists(cs): M.append(("grasp v2 25 spf lam0.25 (68 fr)", sc(np.abs(np.load(cs)).astype(np.float32)), tw))
    rois = dict(Rz); rois["_body"] = body
    panel(M, [("pre ~10s", 10.0), ("peak ~27s", 27.4), ("late ~120s", 120.0)], rois, tw,
          f"{P.OUT}/figures/tofts_vs_patlak_{sim}.png",
          f"phantom xcat {sim}, z15, vs truth. all on the same 5 of 7 angles/frame, one global truth scale, 68-frame window (25 spf). nik seed 0 shown; tables = 3 seeds.")

def invivo(Z):
    sys.path.insert(0, B); import consolidated as C
    ctx = C.slice_ctx(Z); md = np.load(f"{B}/step2_slice{Z}.npz"); mf = md["mf"]; tmf = md["tmf"]; rois = dict(ctx["rois"]); body = ctx["BODY"]; TA = 375.0
    mfv = np.transpose(mf, (1, 2, 0)).astype(np.float32)
    def ft(nt): e = np.linspace(0, TA, nt+1); return 0.5*(e[:-1]+e[1:])
    def refwin(nt):
        e = np.linspace(0, TA, nt+1); o = np.zeros((mf.shape[1], mf.shape[2], nt), np.float32)
        for g in range(nt):
            m = (tmf >= e[g]) & (tmf < e[g+1])
            if not m.any(): m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf-0.5*(e[g]+e[g+1])))] = True
            o[:, :, g] = mf[m].mean(0)
        return o
    def sc(v): R = refwin(v.shape[-1]); return v * (np.sum(v[body]*R[body]) / (np.sum(v[body]**2) + 1e-12))
    M = [("model-free nufft ref (31-spoke window)", mfv, tmf)]
    for arm in ("patlak", "tofts"):
        v = np.abs(np.load(f"{RES}/invivo/{arm}_sl{Z}_s0/nik_slice_{Z:02d}_cplx.npy")).astype(np.float32); M.append((f"{arm} nik s0 (keep_f25, 488 sp, rank {3 if arm=='patlak' else 12})", sc(v), ft(v.shape[-1])))
    g = f"/scratch/rnga/vvpshenov/grasp_v2/results_grasp_v2/gv2_slice{Z}_f25.npy"
    if os.path.exists(g): v = np.abs(np.load(g)).astype(np.float32); M.append(("grasp v2 f25 (488 sp, 122 fr, lam0.25)", sc(v), ft(v.shape[-1])))
    rois["_body"] = body
    panel(M, [("pre ~20s", 20.0), ("peak ~65s", 65.0), ("late ~250s", 250.0)], rois, tmf,
          f"{B}/results/realdata_nik_vs_cs_figures/figures/tofts_vs_patlak_sl{Z}.png",
          f"in vivo slice {Z}, no truth. reference = model-free nufft. nik arms on keep_f25 (488 of 1710 spokes), val v%10==8 early stop; grasp v2 on the same 488 spokes. one global scale vs model-free, baseline-subtracted curves. seed 0 shown; tables = 3 seeds.",
          curve_t=lambda c: c - np.median(c[:8]))

if __name__ == "__main__":
    for s in ("nomotion", "motion"): phantom(s)
    for Z in (18, 19, 21): invivo(Z)
    print("TOFTS_FIGS_DONE")
