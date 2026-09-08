"""Double-check the SENSE-A result: render sub16 (k-space NIK, 2x) AND SENSE-A (image-INR from its
checkpoint) in ONE script, score BOTH with the identical function (no hardcoded baseline), and save a
visual (truth | sub16 | senseA + difference + ROI curves). Confirms whether SENSE-A < sub16 is real
(smoother recon) or a scoring/frame artifact."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_pipeline as P, xph_common as X, recon_asserts as RA
from sense_forward_train import build_image_inr
from masked_metrics import haarpsi_masked, ssim_masked
dev = torch.device("cuda")
d = P.data(); tq = d["times"]; F = len(tq); RO = d["b1"].shape[0]; C = d["b1"].shape[-1]
body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"]); rv = float(Tr[body].max()-Tr[body].min()); Ttime = float(tq.max())
aor = Tr[Rz["aorta"]].mean(0); pk = int(np.argmax(aor)); mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev)
rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1))

def score(rec, name):
    s = np.sum(rec[body]*Tr[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s
    nrmse = float(np.mean([np.sqrt(np.mean((rec[:, :, t][body]-Tr[:, :, t][body])**2))/rv for t in range(F)]))
    pkv = float(Tr[body].max()); mse = float(np.mean([((rec[:, :, t][body]-Tr[:, :, t][body])**2).mean() for t in range(0, F, 2)]))
    psnr = 10*np.log10(pkv**2/(mse+1e-20)); hs, ss = [], []
    for t in range(0, F, 2):
        vmax = float(np.percentile(Tr[:, :, t][body], 99.5))
        pt = torch.from_numpy(np.clip(rec[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev); rt = torch.from_numpy(np.clip(Tr[:, :, t]/(vmax+1e-12), 0, 1)[None, None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt, rt, mt, data_range=1.0).cpu())); ss.append(float(ssim_masked(pt, rt, mt, data_range=1.0).cpu()))
    cur = {nm: (rec[Rz[nm]].mean(0), Tr[Rz[nm]].mean(0), float(np.linalg.norm(rec[Rz[nm]].mean(0)-Tr[Rz[nm]].mean(0))/(np.linalg.norm(Tr[Rz[nm]].mean(0))+1e-12))) for nm in ("aorta", "cortex", "medulla")}
    print(f"{name:12s} PSNR {psnr:6.2f} SSIM {ss and np.mean(ss):.4f} HaarPSI {np.mean(hs):.4f} NRMSE {nrmse:.4f} | aortaC {cur['aorta'][2]:.3f} cortexC {cur['cortex'][2]:.3f} medullaC {cur['medulla'][2]:.3f}", flush=True)
    return rec, cur

# sub16 (k-space NIK, 2x oversampled render)
P.OVERSAMPLE = 2
m16 = P.make_model_g("wire_ff_subspace", 768, P.FIX["k_sigma"], 0, C, dev, rank=16, warmstart=False)
m16.load_state_dict(torch.load(f"{P.OUT}/checkpoints/sub16_w768_s0/ck_24000.pt", map_location=dev, weights_only=False)["state_dict"]); m16.eval()
with torch.no_grad(): dyn16 = P.reconstruct_g(m16, nz=P.build_train(dev)[4], t_query_s=tq, dev=dev)
rec16 = np.abs(np.stack([rot(dyn16[:, :, t]) for t in range(F)], -1))
r16, c16 = score(rec16, "sub16(2x)")

# SENSE-A (image INR, w0=10 s0=5 imsig=2), from checkpoint
mA = build_image_inr(768, 16, 0, 10.0, 5.0, 2.0, 128, dev)
mA.load_state_dict(torch.load(f"{P.OUT}/checkpoints/senseA_f10_r16_s0/ck_final.pt", map_location=dev, weights_only=False)["state_dict"]); mA.eval()
ic = (np.stack(np.meshgrid((np.arange(RO)-RO//2)/(RO//2), (np.arange(RO)-RO//2)/(RO//2), indexing="ij"), -1)).reshape(-1, 2)
icoords = torch.tensor(ic, dtype=torch.float32, device=dev)
with torch.no_grad():
    amp = mA.amplitudes(icoords, torch.zeros(icoords.shape[0], dtype=torch.long, device=dev)); cf = amp[:, :, 0]+1j*amp[:, :, 1]
    Phi = mA.basis(torch.tensor([2*tq[t]/Ttime-1 for t in range(F)], dtype=torch.float32, device=dev))[:, :, 0].real.float()
    dynA = torch.stack([(cf @ Phi[t].to(cf.dtype)).reshape(RO, RO) for t in range(F)], -1).cpu().numpy()
D4 = {"id": lambda x: x, "rot180": lambda x: np.rot90(x, 2), "fliplr": lambda x: x[:, ::-1], "flipud": lambda x: x[::-1, :], "T": lambda x: x.T, "antiT": lambda x: x[::-1, ::-1].T}
mfA = np.abs(dynA).mean(-1); trm = Tr.mean(-1); best = (-2, "id", (0, 0))
for nm, fn in D4.items():
    for dyi in range(-3, 4):
        for dxi in range(-3, 4):
            gg = np.roll(fn(mfA), (dyi, dxi), (0, 1)); cc = float(np.corrcoef(np.abs(gg)[body], trm[body])[0, 1])
            if cc > best[0]: best = (cc, nm, (dyi, dxi))
cc, nm, (dyi, dxi) = best; print(f"senseA registration: D4={nm} shift=({dyi},{dxi}) corr={cc:.3f}", flush=True)
fn = D4[nm]; recA = np.abs(np.stack([np.roll(fn(dynA[:, :, t]), (dyi, dxi), (0, 1)) for t in range(F)], -1))
rA, cA = score(recA, "senseA_w10")

# ---- visual: peak frame truth | sub16 | senseA + diff maps + curves ----
ys, xs = np.where(body); y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1; crop = lambda im: im[y0:y1, x0:x1]
vmax = float(np.percentile(Tr[:, :, pk][body], 99.5))
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
for a, l, im in zip(ax[0], ["truth", "sub16 (2x)", "senseA w10"], [Tr[:, :, pk], r16[:, :, pk], rA[:, :, pk]]):
    a.imshow(crop(im), cmap="gray", vmin=0, vmax=vmax); a.set_title(l, fontsize=11); a.axis("off")
ax[1, 0].imshow(crop(np.abs(r16[:, :, pk]-Tr[:, :, pk])), cmap="magma", vmin=0, vmax=vmax*0.4); ax[1, 0].set_title("|sub16 - truth|", fontsize=10); ax[1, 0].axis("off")
ax[1, 1].imshow(crop(np.abs(rA[:, :, pk]-Tr[:, :, pk])), cmap="magma", vmin=0, vmax=vmax*0.4); ax[1, 1].set_title("|senseA - truth|", fontsize=10); ax[1, 1].axis("off")
axc = ax[1, 2]
for nm, col in [("aorta", "C3"), ("cortex", "C2"), ("medulla", "C0")]:
    axc.plot(tq, cA[nm][1], col+"-", lw=1.5, alpha=.5); axc.plot(tq, c16[nm][0], col+"--", lw=1); axc.plot(tq, cA[nm][0], col+"-", lw=1.2)
axc.set_title("curves: truth(faint) sub16(--) senseA(-)", fontsize=9); axc.set_xlabel("s")
fig.suptitle(f"SENSE-A double-check, peak t={tq[pk]:.0f}s", fontsize=12); plt.tight_layout()
fig.savefig(f"{P.OUT}/figures/senseA_compare.png", dpi=120); print("SAVED figures/senseA_compare.png", flush=True)
print("DONE_CMP", flush=True)
