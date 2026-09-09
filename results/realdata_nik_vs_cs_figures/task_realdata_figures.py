"""Regenerate the REAL in-vivo NIK-vs-CS findings as clean saved PNGs (no re-reconstruction):
(1) spoke-fraction frontier, (2) PK inter-slice CoV, (3) temporal denoising (osc + respiratory frac),
(4) real image montage (model-free reference vs NIK vs CS) + aorta curve. From cached JSONs + recons.
IMPORTANT: the real-data reference is a CS/model-free RECONSTRUCTION (no ground truth) -> these measure
consistency / CS-likeness, not accuracy. Labelled as such."""
import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/net/beegfs/users/P101440/DCE_NIK"; GP = "/net/beegfs/users/P101440/grasp_pro_py"
OUT = f"{B}/results/realdata_nik_vs_cs_figures"; os.makedirs(f"{OUT}/figures", exist_ok=True)
def J(p): return json.load(open(p)) if os.path.exists(p) else None

# ---- (1) spoke-fraction frontier (HaarPSI vs spoke fraction, CS vs NIK; ref = CS-f100) ----
hs = J(f"{B}/haarpsi_spoke.json")
if hs:
    rows = sorted(hs.values(), key=lambda r: -r["pct"])
    pct = sorted({r["pct"] for r in hs.values()}, reverse=True)
    nik = [next(r["mean"] for r in hs.values() if r["pct"] == p and r["meth"] == "NIK") for p in pct]
    cs = [next(r["mean"] for r in hs.values() if r["pct"] == p and r["meth"] == "CS") for p in pct]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pct, cs, "o-", label="CS"); ax.plot(pct, nik, "s-", label="NIK")
    ax.set_xlabel("spoke fraction (%)"); ax.set_ylabel("HaarPSI vs CS-f100 (higher=closer to CS)")
    ax.set_title("real in-vivo: spoke-fraction frontier (slice 13)\nreference = CS-f100 recon -> measures CS-likeness, NOT accuracy")
    ax.legend(); ax.invert_xaxis(); fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig1_spoke_frontier.png", dpi=120); plt.close(fig)

# ---- (2) PK inter-slice CoV (lower = more consistent) ----
t2 = J(f"{B}/task2.json")
if t2 and "consistency" in t2:
    con = t2["consistency"]; params = ["ktrans", "vp"]; rois = ["aorta", "cortex", "medulla", "liver"]; meths = ["NIK_F0", "NIK_F2", "CS_fit"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for pi, par in enumerate(params):
        x = np.arange(len(rois)); wdt = 0.25
        for mi, m in enumerate(meths):
            vals = [con.get(f"{par}_{r}", {}).get(m, np.nan) for r in rois]
            ax[pi].bar(x + (mi-1)*wdt, vals, wdt, label=m)
        ax[pi].set_xticks(x); ax[pi].set_xticklabels(rois); ax[pi].set_title(f"{par} inter-slice CoV (%)  (lower=better)"); ax[pi].set_ylabel("CoV %")
    ax[0].legend(); fig.suptitle("real in-vivo: PK-map inter-slice consistency (NIK vs reconstruct-then-fit CS)")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig2_pk_cov.png", dpi=120); plt.close(fig)

# ---- (3) temporal denoising: oscillation amplitude + respiratory fraction ----
tS = J(f"{B}/task_S.json")
if tS:
    slices = [s for s in tS if s in ("18", "19", "21")]; roi = "cortex"
    meths = [m for m in ("ref", "CS_f100", "CS_f25", "NIK_full", "NIK") if any(m in tS[s] for s in slices)]
    def avg(m, q): return np.nanmean([tS[s][m][roi][q] for s in slices if m in tS[s] and roi in tS[s][m]])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); x = np.arange(len(meths))
    ax[0].bar(x, [avg(m, "osc") for m in meths]); ax[0].set_xticks(x); ax[0].set_xticklabels(meths, rotation=30); ax[0].set_title(f"temporal oscillation amplitude ({roi})\nhigher = more frame-to-frame wobble")
    ax[1].bar(x, [avg(m, "respfrac") for m in meths]); ax[1].set_xticks(x); ax[1].set_xticklabels(meths, rotation=30); ax[1].set_title("respiratory-band power fraction\nlow + high osc = broadband NOISE, not physiology")
    fig.suptitle("real in-vivo: is the CS temporal oscillation physiology or noise?"); fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig3_denoising.png", dpi=120); plt.close(fig)

# ---- (4) real image montage + aorta curve (model-free reference vs NIK vs CS) ----
try:
    z = np.load(f"{B}/step2_slice21.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0); tmf = z["tmf"]; body = z["body"]  # [192,192,240]
    nik = np.load(f"{B}/results_batch/full_sl21f25/nik_slice_21.npy")                                                # [192,192,342]
    tn = np.linspace(tmf[0], tmf[-1], nik.shape[2])
    cs = None
    for p in [f"{GP}/results_spoke_cs/cs_slice21_f25.npy", f"{GP}/results_ref/slice_21.npz"]:
        if os.path.exists(p):
            cs = np.load(p) if p.endswith(".npy") else np.abs(np.load(p)["cs_img"]); break
    aorta = np.load(f"{B}/aorta_roi.npy") if os.path.exists(f"{B}/aorta_roi.npy") else (z["rmap"] > 0)
    # montage at 4 physical times
    tqs = [tmf[np.argmin(abs(tmf-t))] for t in (15, 40, 90, 250)]
    # scale-match NIK/CS to the model-free reference (one global scalar on body) for fair display
    def smatch(vol):
        vb = vol[body].mean(); rb = mf[body].mean(); return vol * (rb / (vb + 1e-12))
    methods = [("model-free ref", mf, tmf), ("NIK (f25)", smatch(nik), tn)]
    if cs is not None: methods.append(("CS (f25)", smatch(cs), np.linspace(tmf[0], tmf[-1], cs.shape[2])))
    vmax = np.percentile(mf[body], 99)
    fig, ax = plt.subplots(len(methods), len(tqs), figsize=(2.4*len(tqs), 2.3*len(methods)), squeeze=False)
    for r, (nm, vol, tg) in enumerate(methods):
        for j, t in enumerate(tqs):
            ax[r, j].imshow(vol[:, :, np.argmin(abs(tg-t))], cmap="gray", vmax=vmax); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
        ax[r, 0].set_ylabel(nm, fontsize=9)
    for j, t in enumerate(tqs): ax[0, j].set_title(f"{t:.0f}s", fontsize=8)
    fig.suptitle("real in-vivo slice 21 (no ground truth; model-free = streak-free reference)"); fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig4_realdata_montage.png", dpi=120); plt.close(fig)
    # aorta curve
    fig, ax = plt.subplots(figsize=(7, 4))
    for nm, vol, tg in methods: ax.plot(tg, vol[aorta].mean(0), lw=1.3, label=nm)
    ax.set_xlabel("time (s)"); ax.set_ylabel("aorta ROI mean signal"); ax.set_title("real in-vivo aorta curve (slice 21)"); ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig5_realdata_aorta_curve.png", dpi=120); plt.close(fig)
    print("montage+curve done")
except Exception as e:
    print("montage skipped:", str(e)[:120])
print("REALDATA_FIGS_DONE ->", OUT)
