"""Visual double-check: truth | sub16 | outcoil (40k, from npz) + difference maps + ROI curves.
Scores both with the identical function. Confirms the output-coil win is real (sharper/cleaner)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import xph_pipeline as P, xph_common as X, recon_asserts as RA
from masked_metrics import haarpsi_masked, ssim_masked
dev = torch.device("cuda")
d = P.data(); tq = d["times"]; F = len(tq); RO = d["b1"].shape[0]; C = d["b1"].shape[-1]
body = d["labels"] > 0; Tr = X.truth_at(P.ZI, tq); Rz = X.rois(P.ZI, d["labels"]); rv = float(Tr[body].max()-Tr[body].min())
aor = Tr[Rz["aorta"]].mean(0); pk = int(np.argmax(aor)); mt = torch.from_numpy(body.astype(np.float32))[None, None].to(dev)
rot = lambda im: np.roll(im[::-1, ::-1], (1, 1), axis=(0, 1))
def score(rec, name):
    s = np.sum(rec[body]*Tr[body])/(np.sum(rec[body]**2)+1e-12); rec = rec*s
    nrmse = float(np.mean([np.sqrt(np.mean((rec[:,:,t][body]-Tr[:,:,t][body])**2))/rv for t in range(F)]))
    pkv = float(Tr[body].max()); mse = float(np.mean([((rec[:,:,t][body]-Tr[:,:,t][body])**2).mean() for t in range(0,F,2)]))
    psnr = 10*np.log10(pkv**2/(mse+1e-20)); hs, ss = [], []
    for t in range(0,F,2):
        vmax=float(np.percentile(Tr[:,:,t][body],99.5))
        pt=torch.from_numpy(np.clip(rec[:,:,t]/(vmax+1e-12),0,1)[None,None]).float().to(dev); rt=torch.from_numpy(np.clip(Tr[:,:,t]/(vmax+1e-12),0,1)[None,None]).float().to(dev)
        hs.append(float(haarpsi_masked(pt,rt,mt,data_range=1.0).cpu())); ss.append(float(ssim_masked(pt,rt,mt,data_range=1.0).cpu()))
    cur={nm:(rec[Rz[nm]].mean(0), Tr[Rz[nm]].mean(0)) for nm in ("aorta","cortex","medulla")}
    print(f"{name:12s} PSNR {psnr:6.2f} SSIM {np.mean(ss):.4f} HaarPSI {np.mean(hs):.4f} NRMSE {nrmse:.4f}", flush=True)
    return rec, cur
# sub16 render
P.OVERSAMPLE=2; m16=P.make_model_g("wire_ff_subspace",768,P.FIX["k_sigma"],0,C,dev,rank=16,warmstart=False)
m16.load_state_dict(torch.load(f"{P.OUT}/checkpoints/sub16_w768_s0/ck_24000.pt",map_location=dev,weights_only=False)["state_dict"]); m16.eval()
with torch.no_grad(): dyn16=P.reconstruct_g(m16,nz=P.build_train(dev)[4],t_query_s=tq,dev=dev)
r16,c16=score(np.abs(np.stack([rot(dyn16[:,:,t]) for t in range(F)],-1)),"sub16(2x)")
rA,cA=score(np.load(f"{P.OUT}/arrays/outcoil_r16_s0.npz")["rec"],"outcoil(40k)")
ys,xs=np.where(body); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1; crop=lambda im: im[y0:y1,x0:x1]
vmax=float(np.percentile(Tr[:,:,pk][body],99.5))
fig,ax=plt.subplots(2,3,figsize=(12,8))
for a,l,im in zip(ax[0],["truth","sub16 (input-coil)","outcoil (output-coil)"],[Tr[:,:,pk],r16[:,:,pk],rA[:,:,pk]]):
    a.imshow(crop(im),cmap="gray",vmin=0,vmax=vmax); a.set_title(l,fontsize=11); a.axis("off")
ax[1,0].imshow(crop(np.abs(r16[:,:,pk]-Tr[:,:,pk])),cmap="magma",vmin=0,vmax=vmax*0.4); ax[1,0].set_title("|sub16 - truth|",fontsize=10); ax[1,0].axis("off")
ax[1,1].imshow(crop(np.abs(rA[:,:,pk]-Tr[:,:,pk])),cmap="magma",vmin=0,vmax=vmax*0.4); ax[1,1].set_title("|outcoil - truth|",fontsize=10); ax[1,1].axis("off")
axc=ax[1,2]
for nm,col in [("aorta","C3"),("cortex","C2"),("medulla","C0")]:
    axc.plot(tq,cA[nm][1],col+"-",lw=1.6,alpha=.45); axc.plot(tq,c16[nm][0],col+"--",lw=1); axc.plot(tq,cA[nm][0],col+"-",lw=1.3)
axc.set_title("curves: truth(faint) sub16(--) outcoil(-)",fontsize=9); axc.set_xlabel("s")
fig.suptitle(f"coil-as-output double-check, peak t={tq[pk]:.0f}s",fontsize=12); plt.tight_layout()
fig.savefig(f"{P.OUT}/figures/outcoil_compare.png",dpi=120); print("SAVED figures/outcoil_compare.png",flush=True); print("DONE_CMP",flush=True)
