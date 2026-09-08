"""TASK 1A: matched CS-f100 reference sensitivity audit. Changes ONLY the full-data reference.
Freezes the current magnitude-first coefficient extraction (verbatim from task2_* scripts).
Compares F0/F2/CS-f25 against (A) old 342-frame CS-f100 (cs_img/recon_slice) and (B) matched
122-frame CS-f100 (cs_slice_f100, spoke-sweep). Also compares the two references directly.
No retraining, no model/extraction change. out: results/task1a_matched_reference/."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os, sys, csv, json, torch, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); sys.path.insert(0, ".")
import consolidated as C
from masked_metrics import haarpsi_masked
D = "/scratch/rnga/vvpshenov/DCE_NIK"; REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
CSD = "/scratch/rnga/vvpshenov/grasp_pro_py/results_spoke_cs"; TA = 375.0; SLICES = [18, 19, 21]
OUT = f"{D}/results/task1a_matched_reference"
for sub in ["figures", "arrays", "logs"]: os.makedirs(f"{OUT}/{sub}", exist_ok=True)

# ---------- FROZEN current magnitude-first coefficient extraction (verbatim) ----------
def bpinv(Z, nt):
    az = np.load(f"{D}/aif_slice{Z}.npz"); tg = np.linspace(0, TA, nt)              # current convention (inferred grid)
    aif = np.interp(tg, np.asarray(az["tC"]), np.asarray(az["aif_frame"])); aif /= (aif.max() + 1e-9)
    iaif = np.concatenate([[0], np.cumsum(0.5 * (aif[1:] + aif[:-1]) * np.diff(tg))]); iaif /= (iaif.max() + 1e-9)
    return np.linalg.pinv(np.stack([aif, iaif, np.ones_like(aif)], 1))
def coeffs(mag, Z):                                                                 # (AIF-coeff |A0|, integrated-AIF-coeff |A1|)
    amp = np.tensordot(mag, bpinv(Z, mag.shape[-1]).T, axes=([2], [0]))
    return np.abs(amp[..., 0]), np.abs(amp[..., 1])
def L(p): return np.abs(np.load(p)).astype(np.float32) if os.path.exists(p) else None

# ---------- frozen metric normalization (reference 99.5-pct, shared for both images) ----------
def tb(a, v): return torch.from_numpy(np.clip(a / (v + 1e-9), 0, 1)[None, None]).float()
def met(a, ref, mask):
    v = float(np.percentile(ref[mask], 99.5)); mt = torch.from_numpy(mask.astype(np.float32))[None, None]
    haar = float(haarpsi_masked(tb(a, v), tb(ref, v), mt, data_range=1.0))
    corr = float(np.corrcoef(a[mask], ref[mask])[0, 1])
    sc = float((a[mask] @ ref[mask]) / (a[mask] @ a[mask] + 1e-12)); mse = float(((a[mask] * sc - ref[mask]) ** 2).mean())
    psnr = float(10 * np.log10(float(ref[mask].max()) ** 2 / (mse + 1e-20)))
    return dict(haarpsi=haar, corr=corr, psnr=psnr, norm_pct=99.5, norm_val=v)

# ---------- source paths ----------
def paths(Z):
    return {"F0_f25": f"{D}/results_batch/pk_f0_sl{Z}/nik_slice_{Z}_cplx.npy",
            "F2_f25": f"{D}/results_batch/pk_f2_sl{Z}/nik_slice_{Z}_cplx.npy",
            "CS_f25": f"{CSD}/cs_slice{Z:02d}_f25.npy",
            "mCS_f100": f"{CSD}/cs_slice{Z:02d}_f100.npy",              # matched 122-frame
            "oCS_f100": f"{REF}/slice_{Z:02d}.npz"}                      # old 342-frame (cs_img)

# ---------- input manifest ----------
man = []
for Z in SLICES:
    P = paths(Z)
    for name, p in P.items():
        if name == "oCS_f100":
            a = np.abs(np.load(p)["cs_img"]); src = "precompute_ref.recon_slice"; frac = "f100"; binning = "recon_slice 342-frame"
        else:
            a = np.abs(np.load(p)) if os.path.exists(p) else None
            src = {"F0_f25": "wire_ff_patlak F0", "F2_f25": "wire_ff_patlak F2", "CS_f25": "cs_spoke_sweep", "mCS_f100": "cs_spoke_sweep"}[name]
            frac = "f25" if "f25" in name else "f100"
            binning = "NIK 342-frame render" if name.startswith("F") else "cs_spoke_sweep 122-frame (NLINE=14)"
        man.append(dict(slice=Z, name=name, path=p, shape=None if a is None else str(a.shape),
                        n_frames=None if a is None else a.shape[-1], source=src, spoke_fraction=frac,
                        binning=binning, timestamps_stored="no (inferred via linspace)", temporal_start_s=0.0, temporal_end_s=TA))
with open(f"{OUT}/input_manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(man[0].keys())); w.writeheader(); [w.writerow(r) for r in man]

# ---------- extract coeff maps + masks ----------
MAP = {}; MASKS = {}
for Z in SLICES:
    ctx = C.slice_ctx(Z); body = ctx["BODY"]; rois = ctx["rois"]
    organ = np.zeros_like(body)
    for r in ["aorta", "cortex", "medulla"]:
        m = rois.get(r)
        if m is not None: organ |= m
    organ = ndi.binary_dilation(organ, iterations=3) & body                          # same construction as previous analysis
    MASKS[Z] = {"body": body, "organ": organ}
    P = paths(Z); MAP[Z] = {}
    for name, p in P.items():
        mag = np.abs(np.load(p)["cs_img"]).astype(np.float32) if name == "oCS_f100" else L(p)
        a0, a1 = coeffs(mag, Z)
        MAP[Z][name] = {"AIF": a0, "intAIF": a1}
        np.save(f"{OUT}/arrays/coeff_{name}_sl{Z}_AIF.npy", a0)
        np.save(f"{OUT}/arrays/coeff_{name}_sl{Z}_intAIF.npy", a1)

# ---------- metrics: comparison A (old ref) and B (matched ref) ----------
rows = []
for Z in SLICES:
    for comp, refname in [("A_old", "oCS_f100"), ("B_matched", "mCS_f100")]:
        for method in ["F0_f25", "F2_f25", "CS_f25"]:
            for mask in ["body", "organ"]:
                for coeff in ["AIF", "intAIF"]:
                    a = MAP[Z][method][coeff]; ref = MAP[Z][refname][coeff]; mk = MASKS[Z][mask]
                    m = met(a, ref, mk)
                    rows.append(dict(slice=Z, comparison=comp, reference=refname, method=method, mask=mask,
                                     coeff_map=coeff, mask_vox=int(mk.sum()), **{k: m[k] for k in ("haarpsi", "corr", "psnr")}))
    # reference-direct: old vs matched
    for mask in ["body", "organ"]:
        for coeff in ["AIF", "intAIF"]:
            a = MAP[Z]["mCS_f100"][coeff]; ref = MAP[Z]["oCS_f100"][coeff]; mk = MASKS[Z][mask]
            m = met(a, ref, mk); nrmse = float(np.sqrt(np.mean((a[mk] - ref[mk]) ** 2)) / (ref[mk].max() - ref[mk].min() + 1e-9))
            rows.append(dict(slice=Z, comparison="REF_old_vs_matched", reference="oCS_f100", method="mCS_f100", mask=mask,
                             coeff_map=coeff, mask_vox=int(mk.sum()), haarpsi=m["haarpsi"], corr=m["corr"], psnr=m["psnr"]))
with open(f"{OUT}/metrics_per_slice.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

# ---------- summary (mean across slices) ----------
summ = {}
for r in rows:
    key = (r["comparison"], r["method"], r["mask"], r["coeff_map"])
    summ.setdefault(key, []).append(r)
with open(f"{OUT}/metrics_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["comparison", "method", "mask", "coeff_map", "haarpsi_mean", "corr_mean", "psnr_mean", "n_slices"])
    for key, rs in sorted(summ.items()):
        w.writerow(list(key) + [round(np.mean([x["haarpsi"] for x in rs]), 4), round(np.mean([x["corr"] for x in rs]), 4),
                                round(np.mean([x["psnr"] for x in rs]), 2), len(rs)])

# ---------- figures ----------
def fig_refcompare(coeff, tag):
    fig, ax = plt.subplots(len(SLICES), 4, figsize=(16, 4 * len(SLICES)))
    for i, Z in enumerate(SLICES):
        old = MAP[Z]["oCS_f100"][coeff]; mat = MAP[Z]["mCS_f100"][coeff]; vmax = np.percentile(mat, 99)
        d = old - mat
        for j, (img, ttl, cmap, vmx, vmn) in enumerate([
            (old, "old CS-f100 (342f)", "viridis", vmax, 0), (mat, "matched CS-f100 (122f)", "viridis", vmax, 0),
            (d, "signed diff (old-matched)", "bwr", np.percentile(np.abs(d), 99), -np.percentile(np.abs(d), 99)),
            (np.abs(d), "abs diff", "inferno", np.percentile(np.abs(d), 99), 0)]):
            im = ax[i, j].imshow(np.rot90(img), cmap=cmap, vmax=vmx, vmin=vmn, interpolation="nearest")
            ax[i, j].axis("off"); ax[i, j].set_title(f"sl{Z} {ttl}", fontsize=9); fig.colorbar(im, ax=ax[i, j], fraction=.046)
    fig.suptitle(f"TASK 1A {tag}: old vs matched CS-f100 reference ({coeff}-coeff map)", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/refcompare_{coeff}.png", dpi=120); plt.close(fig)
fig_refcompare("intAIF", "integrated-AIF")
fig_refcompare("AIF", "AIF")

def fig_methods_vs_matched(coeff):
    fig, ax = plt.subplots(len(SLICES), 4, figsize=(16, 4 * len(SLICES))); cols = ["F0_f25", "F2_f25", "CS_f25", "mCS_f100"]
    for i, Z in enumerate(SLICES):
        vmax = np.percentile(MAP[Z]["mCS_f100"][coeff], 99)
        for j, name in enumerate(cols):
            im = ax[i, j].imshow(np.rot90(MAP[Z][name][coeff]), cmap="viridis", vmax=vmax, vmin=0, interpolation="nearest")
            ax[i, j].axis("off"); ax[i, j].set_title(f"sl{Z} {name}" + (" (ref)" if name == "mCS_f100" else ""), fontsize=9)
    fig.suptitle(f"TASK 1A: methods vs matched CS-f100 ({coeff}-coeff), fixed colour scale/slice", fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/methods_vs_matched_{coeff}.png", dpi=120); plt.close(fig)
fig_methods_vs_matched("intAIF"); fig_methods_vs_matched("AIF")

# per-slice ranking: HaarPSI (organ, intAIF), old vs matched
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for k, comp in enumerate(["A_old", "B_matched"]):
    for method, c in [("F0_f25", "#e8a"), ("F2_f25", "#70c"), ("CS_f25", "#08a")]:
        y = [next(r["haarpsi"] for r in rows if r["slice"] == Z and r["comparison"] == comp and r["method"] == method and r["mask"] == "organ" and r["coeff_map"] == "intAIF") for Z in SLICES]
        ax[k].plot(SLICES, y, "o-", color=c, label=method)
    ax[k].set_title(f"{comp} ref: HaarPSI (organ, intAIF)"); ax[k].set_xlabel("slice"); ax[k].set_xticks(SLICES); ax[k].grid(alpha=.3); ax[k].legend(fontsize=8)
fig.suptitle("TASK 1A per-slice ranking, integrated-AIF coeff, organ mask", fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/per_slice_ranking.png", dpi=120); plt.close(fig)

# ---------- console summary ----------
print("=== metrics_summary (mean across slices) ===")
print(f"{'comparison':>16}{'method':>9}{'mask':>7}{'coeff':>8}{'HaarPSI':>9}{'corr':>7}{'PSNR':>7}")
for key, rs in sorted(summ.items()):
    if key[0] == "REF_old_vs_matched": continue
    print(f"{key[0]:>16}{key[1]:>9}{key[2]:>7}{key[3]:>8}{np.mean([x['haarpsi'] for x in rs]):>9.3f}{np.mean([x['corr'] for x in rs]):>7.3f}{np.mean([x['psnr'] for x in rs]):>7.1f}")
print("\n=== reference direct: old vs matched CS-f100 (agreement + map nRMSE) ===")
for Z in SLICES:
    for coeff in ["intAIF", "AIF"]:
        r = next(x for x in rows if x["comparison"] == "REF_old_vs_matched" and x["slice"] == Z and x["mask"] == "organ" and x["coeff_map"] == coeff)
        print(f"  sl{Z} {coeff:>7} organ: HaarPSI(old,matched) {r['haarpsi']:.3f} corr {r['corr']:.3f} PSNR {r['psnr']:.1f}")
print("\nwrote:", OUT)
