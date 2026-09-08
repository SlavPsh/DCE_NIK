"""TASK 4C aggregation: read the 9 eval_w*.npz, emit all CSVs + 15 figures + pattern flags.
Truth used only here (after checkpoint selection). Task-4 direct result is a fixed descriptive
reference (never used to select a NIK capacity)."""
import warnings; warnings.filterwarnings("ignore")
import os, glob, csv, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "/scratch/rnga/vvpshenov/DCE_NIK/results/task4c_nik_capacity_audit"; FIG = f"{OUT}/figures"
T4 = "/scratch/rnga/vvpshenov/DCE_NIK/results/task4_xcat_nomotion_pilot"
S = np.load(f"{T4}/arrays/sim.npz"); labels = S["labels"]; th = S["theta_true"]; Phi = S["Phi"]; times = S["times"]
body = labels > 0; aorta = labels == 36; LAB = {256: "small", 512: "current", 768: "large"}
# Task-4 direct-discrete f25 fixed descriptive reference
DIRECT_REF = dict(aorta_recovery=0.034, intAIF=0.120, aorta_curve=0.609, AIFmap=0.966, complex=0.130, heldout=0.0005)
TCOL = ["R_aorta", "aorta_curve", "intAIF", "FP", "scale_abs", "scale_phase"]

runs = {}
for p in sorted(glob.glob(f"{OUT}/arrays/eval_w*.npz")):
    d = np.load(p, allow_pickle=True); runs[(int(d["width"]), int(d["seed"]))] = d
print("runs loaded:", sorted(runs.keys()))
def sm(thC):  # global complex scale-match on body
    s = np.vdot(thC[body], th[body]) / (np.vdot(thC[body], thC[body]) + 1e-12); return thC * s

# ---------- CSVs ----------
# checkpoint_metrics (all runs, all checkpoints)
with open(f"{OUT}/checkpoint_metrics.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["run", "width", "seed", "step", "train_nmse", "val_nmse", "val_inner", "val_mid", "val_outer", "R_aorta", "aorta_curve", "intAIF", "FP"])
    for (wd, sd), d in runs.items():
        for i, st in enumerate(d["steps"]):
            vs = d["val_shells"][i]; tr = d["truth"][i]
            w.writerow([f"w{wd}_s{sd}", wd, sd, int(st), f"{d['train_nmse'][i]:.6e}", f"{d['val_nmse'][i]:.6e}", f"{vs[0]:.6e}", f"{vs[1]:.6e}", f"{vs[2]:.6e}", f"{tr[0]:.4f}", f"{tr[1]:.4f}", f"{tr[2]:.4f}", f"{tr[3]:.4f}"])
# best_checkpoint_summary + test_metrics + truth_metrics + overfitting_diagnostics
bc = open(f"{OUT}/best_checkpoint_summary.csv", "w", newline=""); bcw = csv.writer(bc)
bcw.writerow(["run", "width", "seed", "best_step", "final_step", "val_nmse_best", "val_nmse_final", "test_nmse_best", "test_nmse_final", "R_aorta_best", "R_aorta_final", "FP_best", "FP_final"])
tmc = open(f"{OUT}/test_metrics.csv", "w", newline=""); tmw = csv.writer(tmc)
tmw.writerow(["run", "width", "seed", "checkpoint", "test_nmse", "test_inner", "test_mid", "test_outer"])
trc = open(f"{OUT}/truth_metrics.csv", "w", newline=""); trw = csv.writer(trc)
trw.writerow(["run", "width", "seed", "checkpoint", "R_aorta", "aorta_curve", "intAIF", "FP"])
ofc = open(f"{OUT}/overfitting_diagnostics.csv", "w", newline=""); ofw = csv.writer(ofc)
ofw.writerow(["run", "width", "seed", "val_min_step", "final_is_last", "aorta_rises_after_valmin", "FP_rises_late", "final_worse_than_best_val", "final_worse_than_best_test"])
for (wd, sd), d in runs.items():
    bi = int(d["best_idx"]); steps = d["steps"]; val = d["val_nmse"]; tru = d["truth"]
    tb, tf = d["test_best"], d["test_final"]
    bcw.writerow([f"w{wd}_s{sd}", wd, sd, int(d["best_step"]), int(d["final_step"]), f"{val[bi]:.6e}", f"{val[-1]:.6e}", f"{tb[0]:.6e}", f"{tf[0]:.6e}", f"{tru[bi,0]:.4f}", f"{tru[-1,0]:.4f}", f"{tru[bi,3]:.4f}", f"{tru[-1,3]:.4f}"])
    for tag, tv in [("best", tb), ("final", tf)]: tmw.writerow([f"w{wd}_s{sd}", wd, sd, tag, f"{tv[0]:.6e}", f"{tv[1]:.6e}", f"{tv[2]:.6e}", f"{tv[3]:.6e}"])
    for tag, i in [("best", bi), ("final", len(steps) - 1)]: trw.writerow([f"w{wd}_s{sd}", wd, sd, tag, f"{tru[i,0]:.4f}", f"{tru[i,1]:.4f}", f"{tru[i,2]:.4f}", f"{tru[i,3]:.4f}"])
    aorta_after = bool((tru[bi:, 0].max() - tru[bi, 0]) > 0.02)         # aorta recovery rises after val-min
    fp_late = bool(tru[-1, 3] > tru[bi, 3] * 1.1)                       # FP rises from best to final
    ofw.writerow([f"w{wd}_s{sd}", wd, sd, int(d["best_step"]), bool(bi == len(steps) - 1), aorta_after, fp_late, bool(val[-1] > val[bi] * 1.01), bool(tf[0] > tb[0] * 1.01)])
for h in (bc, tmc, trc, ofc): h.close()
# seed_stability (per capacity, over seeds)
with open(f"{OUT}/seed_stability.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["capacity", "width", "n_seeds", "test_nmse_mean", "test_nmse_std", "test_outer_mean", "test_outer_std", "R_aorta_mean", "R_aorta_std", "aorta_curve_mean", "aorta_curve_std", "intAIF_mean", "intAIF_std", "FP_mean", "FP_std", "best_step_mean", "best_step_std"])
    for wd in (256, 512, 768):
        seeds = [(s, d) for (ww, s), d in runs.items() if ww == wd]
        if not seeds: continue
        tn = [d["test_best"][0] for _, d in seeds]; to = [d["test_best"][3] for _, d in seeds]
        ra = [d["truth"][int(d["best_idx"]), 0] for _, d in seeds]; ac = [d["truth"][int(d["best_idx"]), 1] for _, d in seeds]
        ia = [d["truth"][int(d["best_idx"]), 2] for _, d in seeds]; fp = [d["truth"][int(d["best_idx"]), 3] for _, d in seeds]
        bs = [d["best_step"] for _, d in seeds]
        def ms(x): return f"{np.mean(x):.4f}", f"{np.std(x):.4f}"
        w.writerow([LAB[wd], wd, len(seeds), *ms(tn), *ms(to), *ms(ra), *ms(ac), *ms(ia), *ms(fp), f"{np.mean(bs):.0f}", f"{np.std(bs):.0f}"])
print("wrote CSVs", flush=True)

# ---------- FIGURES ----------
def imsh(ax, im, t, vmax=None, cmap="gray"): ax.imshow(np.abs(im), cmap=cmap, vmax=vmax); ax.set_title(t, fontsize=7); ax.axis("off")
AIFvmax = np.abs(th[..., 0]).max()
order = [(w, s) for w in (256, 512, 768) for s in (0, 1, 2) if (w, s) in runs]
# aorta bbox for zooms
ys, xs = np.where(aorta); y0, y1, x0, x1 = ys.min() - 8, ys.max() + 8, xs.min() - 8, xs.max() + 8
# Fig1 truth AIF
fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2)); imsh(ax, th[..., 0], "truth AIF map", AIFvmax); fig.tight_layout(); fig.savefig(f"{FIG}/fig01_truth_AIF.png", dpi=120); plt.close(fig)
# Fig2/3 best-val + final AIF maps (rows=width, cols=seed)
for tag, key, fn, ti in [("best", "thetaC_best", "fig02_bestval_AIF", "best-val"), ("final", "thetaC_final", "fig03_final_AIF", "final-step")]:
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), squeeze=False)
    for r, wd in enumerate((256, 768) if False else (256, 512, 768)):
        for c, sd in enumerate((0, 1, 2)):
            if (wd, sd) in runs: imsh(ax[r, c], sm(runs[(wd, sd)][key])[..., 0], f"w{wd} s{sd}", AIFvmax)
            else: ax[r, c].axis("off")
    fig.suptitle(f"AIF map ({ti})  rows=width[256,512,768] cols=seed[0,1,2]"); fig.tight_layout(); fig.savefig(f"{FIG}/{fn}.png", dpi=110); plt.close(fig)
# Fig4 aorta zooms (best-val), identical scale
fig, ax = plt.subplots(3, 3, figsize=(9, 9), squeeze=False)
for r, wd in enumerate((256, 512, 768)):
    for c, sd in enumerate((0, 1, 2)):
        if (wd, sd) in runs: imsh(ax[r, c], sm(runs[(wd, sd)]["thetaC_best"])[y0:y1, x0:x1, 0], f"w{wd} s{sd}", AIFvmax)
        else: ax[r, c].axis("off")
fig.suptitle("aorta AIF zoom (best-val, identical scale)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig04_aorta_zoom.png", dpi=120); plt.close(fig)
# Fig5 signed + abs AIF diff (best-val - truth) for seed 0 each width
fig, ax = plt.subplots(3, 2, figsize=(7, 9), squeeze=False)
for r, wd in enumerate((256, 512, 768)):
    if (wd, 0) not in runs: continue
    diff = sm(runs[(wd, 0)]["thetaC_best"])[..., 0] - th[..., 0]
    ax[r, 0].imshow(np.real(diff), cmap="RdBu", vmin=-AIFvmax, vmax=AIFvmax); ax[r, 0].set_title(f"w{wd} s0 signed(real) diff", fontsize=7); ax[r, 0].axis("off")
    imsh(ax[r, 1], np.abs(diff), f"w{wd} s0 |diff|", AIFvmax)
fig.suptitle("AIF-map difference (best-val - truth)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig05_AIF_diff.png", dpi=110); plt.close(fig)
# Fig6 intAIF maps best-val
fig, ax = plt.subplots(3, 3, figsize=(9, 9), squeeze=False); iv = np.abs(th[..., 1]).max()
for r, wd in enumerate((256, 512, 768)):
    for c, sd in enumerate((0, 1, 2)):
        if (wd, sd) in runs: imsh(ax[r, c], sm(runs[(wd, sd)]["thetaC_best"])[..., 1], f"w{wd} s{sd}", iv)
        else: ax[r, c].axis("off")
fig.suptitle("integrated-AIF map (best-val)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig06_intAIF.png", dpi=110); plt.close(fig)
# Fig7 aorta temporal curves (best-val)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
cg = np.abs(np.einsum("xyr,tr->xyt", th, Phi)[aorta].mean(0))
for c, wd in enumerate((256, 512, 768)):
    ax[c].plot(times, cg, "k-", lw=2, label="truth")
    for sd in (0, 1, 2):
        if (wd, sd) in runs:
            ch = np.abs(np.einsum("xyr,tr->xyt", sm(runs[(wd, sd)]["thetaC_best"]), Phi)[aorta].mean(0)); ax[c].plot(times, ch, lw=1, label=f"s{sd}")
    ax[c].set_title(f"w{wd} aorta curve"); ax[c].set_xlabel("s")
ax[0].legend(fontsize=7); fig.suptitle("aorta temporal curve (best-val)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig07_aorta_curves.png", dpi=110); plt.close(fig)
# Fig8 train/val NMSE curves
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for c, wd in enumerate((256, 512, 768)):
    for sd in (0, 1, 2):
        if (wd, sd) in runs:
            d = runs[(wd, sd)]; ax[c].semilogy(d["steps"], d["train_nmse"], "--", lw=1, label=f"s{sd} train"); ax[c].semilogy(d["steps"], d["val_nmse"], "-", lw=1, label=f"s{sd} val")
            ax[c].axvline(d["best_step"], color="gray", ls=":", lw=0.5)
    ax[c].set_title(f"w{wd} k-NMSE"); ax[c].set_xlabel("step")
ax[0].legend(fontsize=6); fig.suptitle("train/val k-space NMSE vs step (dotted=best-val step)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig08_nmse_curves.png", dpi=110); plt.close(fig)
# Fig9 aorta recovery vs step ; Fig10 FP vs step
for col, name, fn, ti in [(0, "R_aorta", "fig09_aorta_vs_step", "aorta recovery"), (3, "FP", "fig10_FP_vs_step", "false-positive energy")]:
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for c, wd in enumerate((256, 512, 768)):
        for sd in (0, 1, 2):
            if (wd, sd) in runs:
                d = runs[(wd, sd)]; ax[c].plot(d["steps"], d["truth"][:, col], lw=1, label=f"s{sd}"); ax[c].axvline(d["best_step"], color="gray", ls=":", lw=0.5)
        ax[c].set_title(f"w{wd} {ti}"); ax[c].set_xlabel("step")
    ax[0].legend(fontsize=7); fig.suptitle(f"{ti} vs step (dotted=best-val step; truth used post-hoc only)"); fig.tight_layout(); fig.savefig(f"{FIG}/{fn}.png", dpi=110); plt.close(fig)
# Fig11 test residual vs k radius (shells) at best
fig, ax = plt.subplots(1, 1, figsize=(6, 4)); xsh = [0.15, 0.5, 0.85]
for (wd, sd), d in runs.items(): ax.plot(xsh, d["test_best"][1:], "o-", lw=1, label=f"w{wd}s{sd}")
ax.set_xlabel("normalized k radius (shell centre)"); ax.set_ylabel("test k-NMSE"); ax.set_title("Fig11 test residual vs k-radius (best-val)"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{FIG}/fig11_test_vs_kradius.png", dpi=110); plt.close(fig)
# Fig12 capacity vs test error ; Fig13 aorta rec vs test outer-k ; Fig14 aorta rec vs FP
def scatter(xf, yf, xl, yl, fn, ti):
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4)); col = {256: "tab:blue", 512: "tab:green", 768: "tab:red"}
    for (wd, sd), d in runs.items(): ax.scatter(xf(d), yf(d), c=col[wd], label=f"w{wd}" if sd == 0 else None, s=40); ax.annotate(f"s{sd}", (xf(d), yf(d)), fontsize=6)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ti); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(f"{FIG}/{fn}.png", dpi=110); plt.close(fig)
scatter(lambda d: int(d["width"]), lambda d: d["test_best"][0], "hidden width", "test k-NMSE (best-val)", "fig12_capacity_vs_test", "Fig12 capacity vs test error")
scatter(lambda d: d["test_best"][3], lambda d: d["truth"][int(d["best_idx"]), 0], "test outer-k NMSE", "aorta recovery", "fig13_aorta_vs_outerk", "Fig13 aorta recovery vs outer-k test error")
scatter(lambda d: d["truth"][int(d["best_idx"]), 3], lambda d: d["truth"][int(d["best_idx"]), 0], "false-positive energy", "aorta recovery", "fig14_aorta_vs_FP", "Fig14 aorta recovery vs false-positive energy")
# Fig15 seed-variability summary (aorta recovery mean+-std by capacity, best vs final)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for k, (col, ti) in enumerate([(0, "aorta recovery"), (3, "FP energy")]):
    for wd in (256, 512, 768):
        seeds = [d for (ww, s), d in runs.items() if ww == wd]
        if not seeds: continue
        vb = [d["truth"][int(d["best_idx"]), col] for d in seeds]; vf = [d["truth"][-1, col] for d in seeds]
        ax[k].errorbar(wd - 10, np.mean(vb), yerr=np.std(vb), fmt="o", color="tab:green", capsize=3, label="best-val" if wd == 256 else None)
        ax[k].errorbar(wd + 10, np.mean(vf), yerr=np.std(vf), fmt="s", color="tab:red", capsize=3, label="final" if wd == 256 else None)
    ax[k].set_xlabel("hidden width"); ax[k].set_title(ti); ax[k].set_xticks([256, 512, 768]); ax[k].legend(fontsize=8)
fig.suptitle("Fig15 seed variability (mean +- std over 3 seeds)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig15_seed_variability.png", dpi=110); plt.close(fig)
print("wrote 15 figures ->", FIG, flush=True); print("AGG_DONE", flush=True)
