"""high-res kidney segmentation from respiratory-GATED (end-expiration) NUFFT references.
boundary = wide post-contrast window (sharp kidney), cortico = corticomedullary window (cortex>>medulla).
masks aligned to the mf grid and applied everywhere. gating used ONLY for segmentation (refs/curves ungated)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, finufft, json, os
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B = "/scratch/rnga/vvpshenov/DCE_NIK"; REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
sh = np.load(f"{REF}/shared.npz"); traj = np.asarray(sh["traj_norm"]).astype(np.complex64)
vt = np.asarray(sh["view_time"]).ravel(); TA = float(sh["TA"]); nx = int(sh["nx"]); bas = int(sh["bas"])
sl = np.load(f"{REF}/slice_21.npz"); kdata = np.asarray(sl["kdata_radial"]).astype(np.complex64); b1 = np.asarray(sl["b1"]).astype(np.complex64)
ncc = kdata.shape[2]; den = np.sum(np.abs(b1) ** 2, 2) + 1e-12
try: SIGN = json.load(open(f"{B}/results_nufft/meta.json"))["sign"]
except Exception: SIGN = 1
ts = vt * TA
sg = np.load(f"{B}/selfgate_slice21.npz"); nav = sg["nav"]; order = sg["order"]
nav_orig = np.empty(len(vt), np.float32); nav_orig[order] = nav                     # nav per original spoke
hist, edges = np.histogram(nav_orig, bins=60); mode = 0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])
expir = np.abs(nav_orig - mode) <= np.percentile(np.abs(nav_orig - mode), 40)       # end-expiration plateau, ~40% of spokes
def win_img(mask):
    idx = np.where(mask)[0]; tr = traj[:, idx]; w = np.maximum(np.abs(tr), 1.0 / nx / 4)
    x = (SIGN * 2 * np.pi * tr.real).ravel().astype(np.float64); y = (SIGN * 2 * np.pi * tr.imag).ravel().astype(np.float64)
    acc = sum(finufft.nufft2d1(x, y, (kdata[:, idx, c] * w).astype(np.complex128).ravel(), (nx, nx), eps=1e-4) * np.conj(b1[:, :, c]) for c in range(ncc))
    s = (nx - bas) // 2; return np.abs(acc / den)[s:s + bas, s:s + bas], len(idx)
post = (ts > 60) & (ts < 375); cm = (ts > 65) & (ts < 100)
bnd, nb = win_img(post & expir); cor, nc_ = win_img(cm & expir); bnd_u, _ = win_img(post)   # gated boundary, gated cortico, UNGATED boundary (compare)
print("gated spokes: boundary %d/%d  cortico %d/%d" % (nb, post.sum(), nc_, cm.sum()))
# orient to mf grid
z = np.load(f"{B}/step2_slice21.npz"); mf = np.abs(z["mf"]).transpose(1, 2, 0); body = z["body"]; mfm = mf.mean(2)
def corr(a, b): a = a.ravel() - a.mean(); b = b.ravel() - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
oris = {"id": lambda x: x, "rot90": lambda x: np.rot90(x), "rot180": lambda x: np.rot90(x, 2), "rot270": lambda x: np.rot90(x, 3),
        "flipud": lambda x: np.flipud(x), "fliplr": lambda x: np.fliplr(x), "T": lambda x: x.T, "Tf": lambda x: np.fliplr(x.T)}
best = max(oris, key=lambda k: corr(oris[k](bnd), mfm)); O = oris[best]
print("orientation to mf: %s (corr %.3f)" % (best, corr(O(bnd), mfm)))
bnd, cor, bnd_u = O(bnd), O(cor), O(bnd_u)
# --- FULL-BEAN kidney on the sharp gated boundary (moderate threshold + close + fill) ---
kseed = body & (bnd > np.percentile(bnd[body], 87)); kseed = ndi.binary_closing(kseed, iterations=2)
lab, n = ndi.label(kseed); cy, cx = ndi.center_of_mass(body); comps = []
for i in range(1, n + 1):
    m = lab == i; s = int(m.sum())
    if not (60 < s < 1500): continue
    yy, xx = ndi.center_of_mass(m)
    if yy < cy - 6: continue                                     # posterior only (drop liver/spleen)
    comps.append((s, ndi.binary_fill_holes(ndi.binary_closing(m, iterations=3)) & body))
comps.sort(key=lambda t: -t[0]); kid = np.zeros_like(body)
for _, m in comps[:2]: kid |= ndi.binary_fill_holes(m)
# cortex = outer rim, medulla = inner core (morphological, per kidney) - robust, no streaky cortico dependence
core = np.zeros_like(body); klab, kn = ndi.label(kid)
for i in range(1, kn + 1): core |= ndi.binary_erosion(klab == i, iterations=2)
medulla = core & kid; cortex = kid & ~core
print("kidneys %d  cortex(rim) %d medulla(core) %d whole %d" % (kn, cortex.sum(), medulla.sum(), kid.sum()))
np.savez(f"{B}/realkid_slice21.npz", cortex=cortex, medulla=medulla, kidney=kid,
         static=(body & (mf.max(2) - mf[:, :, :10].mean(2) < np.percentile((mf.max(2) - mf[:, :, :10].mean(2))[body], 15)) & ~kid))
# overlay: ungated mf vs gated refs + masks
vmax = np.percentile(bnd[body], 99.5)
fig, ax = plt.subplots(1, 4, figsize=(19, 5))
ax[0].imshow(mfm, cmap="gray", vmax=np.percentile(mfm[body], 99.5)); ax[0].set_title("UNGATED model-free mean (blurry)")
ax[1].imshow(bnd_u, cmap="gray", vmax=vmax); ax[1].set_title("ungated post-contrast NUFFT")
ax[2].imshow(bnd, cmap="gray", vmax=vmax); ax[2].set_title(f"GATED post-contrast NUFFT (n={nb})")
ax[3].imshow(cor, cmap="gray", vmax=np.percentile(cor[body], 99.5)); ax[3].set_title(f"GATED corticomedullary (n={nc_})")
for a in ax: a.contour(cortex, [.5], colors="lime", linewidths=1.1); a.contour(medulla, [.5], colors="orange", linewidths=1.1); a.axis("off")
fig.suptitle("real slice 21: gated high-res segmentation (green=cortex, orange=medulla)"); fig.tight_layout()
o = f"{B}/results/realdata_nik_vs_cs_figures/figures/realkid_gated.png"; fig.savefig(o, dpi=115); print("SAVED", o)
