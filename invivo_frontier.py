"""in-vivo spatial-vs-temporal frontier, slice 21: classic GRASP v2 (swept over NLINE) vs NIK.

mirrors the phantom sweep (xph_frontier.py) but there is NO ground truth in vivo, so the
method-neutral model-free NUFFT recon (step2_slice21.npz['mf'], gridding + ramp dcf, no
regularization) stands in for truth. both methods are scored against the SAME reference on the
SAME windows, so the comparison is fair even though absolute values are depressed by mf's noise.

*** LIMITATION, stated up front: mf uses a 31-spoke sliding window, so its EFFECTIVE temporal
resolution is ~6.8 s even though it is sampled every 1.54 s. it therefore CANNOT adjudicate the
fine-temporal regime (<6.8 s/frame) where the phantom's both-axes claim lives. in vivo this
measures the spatial side of the tradeoff, NIK's flatness, and bolus-peak recovery. it is
consistency vs a reference recon, not accuracy. ***

NLINE is matched to the phantom sweep by SECONDS per frame (0.2193 s/spoke here).
out: v2_sweep_invivo/frontier_invivo.json + figures/fig_v2_frontier_invivo.png
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, glob
import numpy as np, torch
sys.path.insert(0, "/scratch/rnga/vvpshenov/DCE_NIK")
from masked_metrics import haarpsi_masked, ssim_masked
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

B = "/scratch/rnga/vvpshenov/DCE_NIK"; GV = "/scratch/rnga/vvpshenov/grasp_v2/results_grasp_v2"
OUTD = f"{B}/v2_sweep_invivo"; FIG = f"{B}/results/realdata_nik_vs_cs_figures/figures"
os.makedirs(OUTD, exist_ok=True)
TA, NTV = 375.0, 1710
SPOKE_S = TA / NTV
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z = np.load(f"{B}/step2_slice21.npz")
mf = np.abs(z["mf"]).transpose(1, 2, 0).astype(np.float32)      # [192,192,240]
tmf = z["tmf"].astype(np.float64); body = z["body"].astype(bool)
ao = np.asarray(np.load(f"{B}/aif_slice21.npz")["ao"]).astype(bool)
mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev)
mf_ao = np.array([mf[..., i][ao].mean() for i in range(mf.shape[-1])])

def frame_times(nt):
    """centre time of each frame for a recon with nt uniform frames over the acquisition."""
    e = np.linspace(0, TA, nt + 1); return 0.5 * (e[:-1] + e[1:])

def ref_windows(nt):
    """model-free reference averaged over the SAME windows the frame integrates."""
    e = np.linspace(0, TA, nt + 1)
    out = np.zeros((mf.shape[0], mf.shape[1], nt), np.float32)
    for g in range(nt):
        m = (tmf >= e[g]) & (tmf < e[g + 1])
        if not m.any():                                          # window finer than mf sampling
            m = np.zeros_like(tmf, bool); m[np.argmin(np.abs(tmf - 0.5 * (e[g] + e[g + 1])))] = True
        out[:, :, g] = mf[:, :, m].mean(2)
    return out

def score(rec, nt):
    R = ref_windows(nt)
    s = np.sum(rec[body] * R[body]) / (np.sum(rec[body] ** 2) + 1e-12)   # single global scale
    rec = rec * s
    hs, ss = [], []
    for t in range(0, nt, max(1, nt // 40)):
        vmax = float(np.percentile(R[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, t] / (vmax + 1e-12), 0, 1)[None, None]).float().to(dev)
        rt = torch.from_numpy(np.clip(R[:, :, t] / (vmax + 1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu()))
        ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))
    tG = frame_times(nt)
    ca = np.array([rec[..., i][ao].mean() for i in range(nt)])
    ci = np.interp(tmf, tG, ca)                                  # onto the mf grid
    def bsub(c): return c - np.median(c[:8])
    cn = float(np.linalg.norm(bsub(ci) - bsub(mf_ao)) / (np.linalg.norm(bsub(mf_ao)) + 1e-12))
    return dict(haarpsi=float(np.mean(hs)), ssim=float(np.mean(ss)), c_aorta=cn,
                aorta_pk_ratio=float(bsub(ci).max() / (bsub(mf_ao).max() + 1e-12)),
                aorta_ttp=float(tmf[np.argmax(bsub(ci))]))

rows = []
# SPOKE MATCHING (this comparison was broken once already, do not regress):
# results_spoke_full_slice21 used train_grasp_nik.py's DEFAULT --subsample-frac 0.7, i.e. NIK saw
# only 70% of spokes while grasp v2 here uses ~100%. that gave grasp 43% more data. the matched run
# trains on keep_f100 (1708 spokes) with NO heldout, the same input grasp gets.
NIK_MATCHED = f"{B}/results_full_sl21_matched/nik_slice_21.npy"
NIK_OLD = f"{B}/results_spoke_full_slice21/nik_slice_21.npy"
if os.path.exists(NIK_MATCHED):
    nik = np.abs(np.load(NIK_MATCHED)).astype(np.float32)
    print(f"NIK: spoke-MATCHED run (keep_f100, 1708 spokes, no heldout)  {nik.shape}")
elif os.environ.get("ALLOW_UNMATCHED") == "1":
    nik = np.abs(np.load(NIK_OLD)).astype(np.float32)
    print("*** WARNING: unmatched NIK (random 70% split). grasp has ~43% more data. NOT a fair frontier. ***")
else:
    raise SystemExit("refusing to run: spoke-matched NIK not found at\n  " + NIK_MATCHED +
                     "\nrun job_full_slice21_matched.sh first, or set ALLOW_UNMATCHED=1 to see the biased version.")
for f in sorted(glob.glob(f"{GV}/gv2_slice21_n*.npy")) + [f"{GV}/gv2_slice21_p05.npy", f"{GV}/gv2_slice21_f100.npy"]:
    if not os.path.exists(f): continue
    bn = os.path.basename(f)
    NL = 5 if "p05" in bn else (14 if "f100" in bn else int(bn.split("_n")[1][:2]))
    rec = np.abs(np.load(f)).astype(np.float32); nt = rec.shape[-1]
    r = score(rec, nt); r.update(method="GRASP-v2", NLINE=NL, frames=nt, dt=NL * SPOKE_S)
    rows.append(r)
    # NIK on the SAME windows: window-average its continuous render
    e = np.linspace(0, TA, nt + 1); tN = frame_times(nik.shape[-1])
    ng = np.stack([nik[:, :, (tN >= e[g]) & (tN < e[g + 1])].mean(2)
                   if ((tN >= e[g]) & (tN < e[g + 1])).any()
                   else nik[:, :, np.argmin(np.abs(tN - 0.5 * (e[g] + e[g + 1])))] for g in range(nt)], -1)
    q = score(ng, nt); q.update(method="NIK", NLINE=NL, frames=nt, dt=NL * SPOKE_S)
    rows.append(q)

rows.sort(key=lambda r: (r["method"], r["NLINE"]))
json.dump(rows, open(f"{OUTD}/frontier_invivo.json", "w"), indent=1)
print("in vivo, slice 21, vs model-free NUFFT reference (consistency, NOT accuracy)")
print("mf effective temporal resolution ~6.8s -> cannot adjudicate <6.8s/frame\n")
print(f"{'method':9} {'NLINE':>6} {'frames':>7} {'s/fr':>6} {'HaarPSI':>8} {'SSIM':>7} {'aortaC':>7} {'pk/mf':>7}")
for r in rows:
    print(f"{r['method']:9} {r['NLINE']:>6} {r['frames']:>7} {r['dt']:>6.2f} {r['haarpsi']:>8.4f} {r['ssim']:>7.4f} {r['c_aorta']:>7.4f} {r['aorta_pk_ratio']:>7.4f}")

g = [r for r in rows if r["method"] == "GRASP-v2"]; n = {r["NLINE"]: r for r in rows if r["method"] == "NIK"}
if g:
    bs = max(g, key=lambda r: r["haarpsi"]); bt = min(g, key=lambda r: r["c_aorta"])
    print(f"\ngrasp v2 best SPATIAL : NLINE={bs['NLINE']} ({bs['dt']:.1f}s) haarpsi {bs['haarpsi']:.4f}")
    print(f"grasp v2 best TEMPORAL: NLINE={bt['NLINE']} ({bt['dt']:.1f}s) aortaC {bt['c_aorta']:.4f}")
    print(f"\n{'NLINE':>6} {'s/fr':>6} | {'haarpsi g':>9} {'haarpsi n':>9} {'d':>7} | {'aortaC g':>8} {'aortaC n':>8} {'d':>7}")
    for r in sorted(g, key=lambda z: z["NLINE"]):
        q = n.get(r["NLINE"])
        if q: print(f"{r['NLINE']:>6} {r['dt']:>6.2f} | {r['haarpsi']:>9.4f} {q['haarpsi']:>9.4f} {q['haarpsi']-r['haarpsi']:>+7.4f}"
                    f" | {r['c_aorta']:>8.4f} {q['c_aorta']:>8.4f} {q['c_aorta']-r['c_aorta']:>+7.4f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
LEG = {"GRASP-v2": "GRASP-v2 classic 2014: MCNUFFT + temporal TV, no subspace, lam=0.25max|x0|, all spokes",
       "NIK": "NIK: wire_ff_res full rank, spoke-matched (keep_f100, 1708 spokes, no heldout)"}
for m, c in (("GRASP-v2", "#c0392b"), ("NIK", "#7c3aed")):
    r = sorted([x for x in rows if x["method"] == m], key=lambda z: z["NLINE"])
    if not r: continue
    ax[0].plot([x["c_aorta"] for x in r], [x["haarpsi"] for x in r], "o-", color=c, label=LEG[m], lw=2, ms=6)
    for x in r: ax[0].annotate(f"{x['NLINE']}", (x["c_aorta"], x["haarpsi"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[1].plot([x["dt"] for x in r], [x["aorta_pk_ratio"] for x in r], "o-", color=c, label=LEG[m], lw=2, ms=6)
ax[0].set_xlabel("aorta curve nrmse vs model-free (lower better)"); ax[0].set_ylabel("haarpsi vs model-free (higher better)")
ax[0].set_title("in vivo frontier, slice 21\nlabels = spokes/frame (NLINE)"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=6.5, loc="lower left")
ax[1].axhline(1.0, color="0.5", ls="--", lw=1); ax[1].set_xlabel("s/frame"); ax[1].set_ylabel("aorta peak / model-free peak")
ax[1].set_title("bolus recovery vs model-free"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=6.5, loc="lower left")
fig.suptitle("in vivo slice 21, NO ground truth. reference = model-free NUFFT (gridding + ramp dcf, "
             "31-spoke sliding window, ~6.8 s effective temporal resolution)\nboth methods spoke-matched: NIK 1708 spokes no heldout, grasp all acquired. consistency, NOT accuracy", fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_v2_frontier_invivo.png", dpi=130)
print(f"\nSAVED {FIG}/fig_v2_frontier_invivo.png")
print("INVIVO_FRONTIER_DONE")
