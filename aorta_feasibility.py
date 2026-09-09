"""Aorta bolus test -- FEASIBILITY v2. Segment the aorta on CLEAN 12-frame CS-100 (bright,
round, central, early-enhancing blob), then show its curve at 12-frame (clean/coarse) AND
187-frame (fine/noisy) + navigator. Question: is there a resolvable sharp bolus, and how
noisy is CS-fine (= NIK's opportunity)? out: aorta_feasibility.png"""
import numpy as np, scipy.ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
A = "/net/beegfs/users/P101440/presentation/assets"; D = "/net/beegfs/users/P101440/DCE_NIK"
GP = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"
clean = np.abs(np.load(f"{A}/arm1_cs100_sl13.npy")).astype(np.float32)      # [x,y,12] clean
fine = np.abs(np.load(f"{A}/arm_temporal_cs100_187.npy")).astype(np.float32)  # [x,y,187]
nC, nF = clean.shape[-1], fine.shape[-1]

# --- navigator (data k=0), independent bolus-timing reference ---
import sys; sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py"); import nik_adapter as NA
from figpath import fig as fpath
sh = NA.load_shared(GP); sl = NA.load_slice(GP, 13); krad = np.asarray(sl["kdata_radial"]); vt = np.asarray(sh["view_time"]).ravel()
c0 = krad.shape[0]//2; dc = np.sqrt((np.abs(krad[c0])**2).sum(-1)); o = np.argsort(vt); ts = vt[o]
navr = np.clip(dc[o], np.percentile(dc,1), np.percentile(dc,99))
nav = np.convolve(np.pad(navr,(40,40),mode="reflect"), np.ones(81)/81, mode="valid")[:len(navr)]

# --- segment aorta on the clean recon: bright + round + central + early-enhancing ---
mean = clean.mean(-1); body = mean > np.quantile(mean, 0.5); H, W = mean.shape
base = clean[..., :2].mean(-1); enh = clean - base[..., None]
ttp = enh.argmax(-1); early = ttp <= 4                                      # peaks in first third
# identify the aorta as a small BRIGHT ROUND BLOB via Laplacian-of-Gaussian (vessel scale ~2.5px),
# then a small fixed disk around its center -- isolates the vessel core without merging neighbors.
cy0, cx0 = ndi.center_of_mass(body)                                        # body centroid
yy, xx = np.ogrid[:H, :W]
central = ((yy-cy0)**2 + (xx-cx0)**2) <= 40**2                             # aorta is central (near centroid)
art = clean[..., 3]                                                        # arterial-phase frame (aorta bright)
log = -ndi.gaussian_laplace(ndi.gaussian_filter(art, 0.6), sigma=2.5)      # strong at bright ~5px blobs
log_m = log * central * (mean > np.quantile(mean[body], 0.85))            # bright + CENTRAL only
sy, sx = np.unravel_index(int(np.argmax(log_m)), log_m.shape)              # aorta center
aorta = ((yy-sy)**2 + (xx-sx)**2) <= 2.4**2                                # small disk (r=2.4)
print(f"aorta ROI: {int(aorta.sum())} vox (LoG blob), center ({sy},{sx})", flush=True)

# liver ROI: one compact disk in the largest HOMOGENEOUS slow-enhancing organ (liver, not bowel).
lm = ndi.uniform_filter(mean, 9); lvar = np.sqrt(np.maximum(ndi.uniform_filter(mean**2, 9) - lm**2, 0))
homog = body & (lvar < np.quantile(lvar[body], 0.35)) & (mean > np.quantile(mean[body], 0.45)) & (ttp >= 5)
homog = ndi.binary_opening(homog, iterations=2)                            # drop specks/thin bowel
ll, nl = ndi.label(homog)
if nl:
    big = ll == (1 + int(np.argmax(ndi.sum(np.ones_like(ll), ll, range(1, nl + 1)))))
    dt = ndi.distance_transform_edt(big); ly, lx = np.unravel_index(int(np.argmax(dt)), dt.shape)  # deepest = liver core
    liver = ((yy - ly) ** 2 + (xx - lx) ** 2) <= 9 ** 2                    # single compact disk
else:
    liver = np.zeros_like(body)
print(f"liver ROI: {int(liver.sum())} vox disk at ({ly},{lx})", flush=True)

def cv(vol, m): return np.array([vol[..., i][m].mean() for i in range(vol.shape[-1])])
def nrm(c, b): return (c - b) / (c.max() - b + 1e-9)
acC, acF = cv(clean, aorta), cv(fine, aorta); lcC = cv(clean, liver)
bC = acC[:2].mean(); bF = acF[:5].mean()
tC = np.arange(nC)/(nC-1)*374; tF = np.arange(nF)/(nF-1)*374; tN = ts*374

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
af = clean[..., 3]; ax[0].imshow(np.rot90(af), cmap="gray", vmin=0, vmax=np.percentile(af,99.5))
ov = np.zeros((H, W, 4)); ov[aorta] = [1,0.15,0.15,1]; ov[liver] = [0.3,0.6,1,0.5]
ax[0].imshow(np.rot90(ov)); ax[0].set_title("clean CS-100 + ROIs (red=aorta, blue=liver)", fontsize=10); ax[0].axis("off")
ax[1].plot(tC, nrm(acC,bC), "s-", color="tab:red", lw=2, ms=5, label="aorta (CS 12-frame, clean)")
ax[1].plot(tC, nrm(lcC,lcC[:2].mean()), "^-", color="tab:blue", lw=1.6, ms=4, label="liver (CS 12-frame)")
ax[1].plot(tN, (nav-nav.min())/(nav.max()-nav.min()), color="0.5", ls="--", lw=1.5, label="navigator (data)")
ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("norm. enh."); ax[1].set_title("clean/coarse: is there a sharp bolus?", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[2].plot(tF, nrm(acF,bF), "-", color="tab:red", lw=1.3, label="aorta (CS 187-frame = 2s, fine)")
ax[2].plot(tC, nrm(acC,bC), "s-", color="darkred", lw=2, ms=5, label="aorta (CS 12-frame)")
ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("norm. enh."); ax[2].set_title("fine CS is noisy = NIK's opportunity", fontsize=10); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
fig.suptitle("Aorta bolus test -- feasibility v2 (slice 13)", fontweight="bold")
fig.tight_layout(); fig.savefig(fpath(f"aorta_feasibility.png"), bbox_inches="tight", dpi=140)
np.save(f"{D}/aorta_roi.npy", aorta); np.save(f"{D}/liver_roi.npy", liver)
print(f"aorta clean upslope {np.gradient(nrm(acC,bC), tC).max():.4f}/s | peak-to-plateau ratio {acC.max()/acC[-3:].mean():.2f}")
print("wrote aorta_feasibility.png")
