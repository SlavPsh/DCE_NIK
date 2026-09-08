"""consolidated figures + matrix + verdicts from report_data.json. adds P3 vp/Ktrans maps
(project the complex recon onto the fixed Patlak columns [AIF, integral AIF] by LS) and their
agreement with a conventional Patlak fit of the CS recon. writes 3 figures + report_matrix.json."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); from figpath import fig as fpath
D = "/scratch/rnga/vvpshenov/DCE_NIK"; REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"; TA = 375.0
rows = json.load(open(f"{D}/report_data.json"))
def get(Z, cfg): return next((r for r in rows if r["slice"] == Z and r["cfg"] == cfg and "spatial" in r), None)

# ============ P1: spatial replication + temporal cortical bias ============
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4)); SL = [18, 19, 20]
x = np.arange(len(SL)); w = 0.35
for j, ruler in enumerate(["rulerA_haarpsi", "rulerB_haarpsi"]):
    cs = [get(Z, "CS")["spatial"][ruler] for Z in SL]; r16 = [get(Z, "R16")["spatial"][ruler] for Z in SL]
    ax[j].bar(x - w / 2, cs, w, label="CS", color="#08a"); ax[j].bar(x + w / 2, r16, w, label="NIK R16", color="#70c")
    ax[j].set_xticks(x); ax[j].set_xticklabels([f"sl{Z}" for Z in SL]); ax[j].legend(fontsize=8); ax[j].grid(alpha=.3, axis="y")
    ax[j].set_title(f"{'ruler A (pre-contrast)' if j==0 else 'ruler B (non-enh mean)'} HaarPSI @f25"); ax[j].set_ylim(0, 1)
# temporal cortex: real vs CS vs NIK-full plateau
csn = [get(Z, "CS")["temporal"]["cortex"]["nrmse"] for Z in SL]; fun = [get(Z, "full")["temporal"]["cortex"]["nrmse"] for Z in SL]
ax[2].bar(x - w / 2, csn, w, label="CS", color="#08a"); ax[2].bar(x + w / 2, fun, w, label="NIK full", color="#e62")
ax[2].set_xticks(x); ax[2].set_xticklabels([f"sl{Z}" for Z in SL]); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")
ax[2].set_title("CORTEX temporal nRMSE vs real (lower=better)")
fig.suptitle("P1 replication: R16 spatial vs CS (2 rulers) + cortex temporal fidelity, slices 18-20", fontweight="bold")
fig.tight_layout(); p1 = fpath("P1_replication.png"); fig.savefig(p1, dpi=135); plt.close(fig)

# ============ P2: rank pareto (slice 21) ============
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
cfgs = ["R8", "R16", "R32", "R64", "full", "PK_F0", "PK_F2", "PK_F4"]
rr = [get(21, c)["rrank"] for c in cfgs]; nom = {"R8": 8, "R16": 16, "R32": 32, "R64": 64, "full": ">64", "PK_F0": 3, "PK_F2": 5, "PK_F4": 7}
ax[0].bar(range(len(cfgs)), rr, color="#70c"); ax[0].set_xticks(range(len(cfgs))); ax[0].set_xticklabels([f"{c}\n(nom {nom[c]})" for c in cfgs], fontsize=8)
ax[0].axhline(5, color="r", ls="--", label="CS K=5"); ax[0].set_ylabel("REALIZED complex rank (SVD)"); ax[0].legend(); ax[0].grid(alpha=.3, axis="y")
ax[0].set_title("rank knob SATURATES: R8-R64 all collapse to realized ~5")
# pareto: spatial ruler A (y, higher better) vs cortex nrmse (x, lower better)
for c in cfgs:
    g = get(21, c); ax[1].scatter(g["temporal"]["cortex"]["nrmse"], g["spatial"]["rulerA_haarpsi"], s=80)
    ax[1].annotate(f"{c}(r{g['rrank']})", (g["temporal"]["cortex"]["nrmse"], g["spatial"]["rulerA_haarpsi"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
cs = get(21, "CS"); ax[1].scatter(cs["temporal"]["cortex"]["nrmse"], cs["spatial"]["rulerA_haarpsi"], s=140, marker="*", color="#08a", label="CS K=5", zorder=5)
ax[1].set_xlabel("cortex temporal nRMSE vs real (better ->)"); ax[1].set_ylabel("ruler A HaarPSI (better ^)"); ax[1].invert_xaxis(); ax[1].legend(); ax[1].grid(alpha=.3)
ax[1].set_title("pareto: does any config sit up-and-RIGHT of CS? (better on both)")
fig.suptitle("P2 rank pareto, slice 21 (realized rank + spatial/temporal tradeoff)", fontweight="bold")
fig.tight_layout(); p2 = fpath("P2_rank_pareto.png"); fig.savefig(p2, dpi=135); plt.close(fig)

# ============ P3: Patlak vp/Ktrans maps + agreement with conventional CS fit ============
az = np.load(f"{D}/aif_slice21.npz"); tC342 = np.linspace(0, TA, 342)
aif342 = np.interp(tC342, np.asarray(az["tC"]), np.asarray(az["aif_frame"])); aif342 = aif342 / (aif342.max() + 1e-9)
iaif342 = np.concatenate([[0], np.cumsum(0.5 * (aif342[1:] + aif342[:-1]) * np.diff(tC342))]); iaif342 /= (iaif342.max() + 1e-9)
Bfix = np.stack([aif342, iaif342, np.ones_like(aif342)], 1)               # [342,3] fixed Patlak columns
Bpinv = np.linalg.pinv(Bfix)
import scipy.ndimage as ndi
sl21 = np.load(f"{REF}/slice_21.npz"); cs100 = np.abs(sl21["cs_img"]).astype(np.float32)
body = cs100.mean(-1) > np.quantile(cs100.mean(-1), 0.55)
# conventional Patlak fit of the CS recon (magnitude), per voxel
cs_amp = np.tensordot(cs100, Bpinv.T, axes=([2], [0]))                    # [x,y,3]  vp,Ktrans,base
vp_cs, kt_cs = np.abs(cs_amp[..., 0]), np.abs(cs_amp[..., 1])
fig, ax = plt.subplots(3, 4, figsize=(15, 10)); rois = None
pkrows = {}
for i, (F, lab) in enumerate([(0, "PK_F0"), (2, "PK_F2"), (4, "PK_F4")]):
    path = f"{D}/results_batch/pk_f{F}_sl21/nik_slice_21_cplx.npy"
    if not os.path.exists(path):
        for a in ax[i]: a.axis("off"); continue
    rec = np.load(path); mag = np.abs(rec)
    amp = np.tensordot(mag, Bpinv.T, axes=([2], [0])); vp, kt = np.abs(amp[..., 0]), np.abs(amp[..., 1])
    ag_vp = float(np.corrcoef(vp[body], vp_cs[body])[0, 1]); ag_kt = float(np.corrcoef(kt[body], kt_cs[body])[0, 1])
    neg = float((amp[..., 1][body] < 0).mean())                          # Patlak Ktrans should be >=0
    pkrows[lab] = dict(agree_vp=ag_vp, agree_ktrans=ag_kt, neg_ktrans_frac=neg)
    v = np.rot90(vp); k = np.rot90(kt)
    ax[i, 0].imshow(v, cmap="magma", vmax=np.percentile(v, 99)); ax[i, 0].set_title(f"{lab} vp", fontsize=9); ax[i, 0].axis("off")
    ax[i, 1].imshow(k, cmap="viridis", vmax=np.percentile(k, 99)); ax[i, 1].set_title(f"{lab} Ktrans", fontsize=9); ax[i, 1].axis("off")
    ax[i, 2].imshow(np.rot90(vp_cs), cmap="magma", vmax=np.percentile(vp_cs, 99)); ax[i, 2].set_title("CS-fit vp", fontsize=9); ax[i, 2].axis("off")
    ax[i, 3].imshow(np.rot90(kt_cs), cmap="viridis", vmax=np.percentile(kt_cs, 99)); ax[i, 3].set_title(f"CS-fit Ktrans\nagree vp {ag_vp:.2f} kt {ag_kt:.2f}", fontsize=9); ax[i, 3].axis("off")
fig.suptitle("P3 Patlak vp/Ktrans maps (NIK) vs conventional Patlak fit of CS, slice 21", fontweight="bold")
fig.tight_layout(); p3 = fpath("P3_patlak_maps.png"); fig.savefig(p3, dpi=130); plt.close(fig)

json.dump(pkrows, open(f"{D}/report_pk.json", "w"), indent=1, default=float)
print("figures:", p1.split("/")[-1], p2.split("/")[-1], p3.split("/")[-1])
print("PK agreement:", json.dumps(pkrows, indent=1, default=lambda x: round(float(x), 3)))
