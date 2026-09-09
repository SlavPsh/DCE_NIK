"""STEP-3 activation comparison: siren vs gabor vs gaussian temporal net, aorta curve + peak-frame images."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RD="results/realdata_nik_vs_cs_figures"; TA=375.0
import os as _os
# reference-method plumbing. defaults = grasp-pro (unchanged). grasp v2:
#   CSD=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2 CSPRE=gv2 TAG=_gv2
_CSD = _os.environ.get("CSD", "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs")
_CSPRE = _os.environ.get("CSPRE", "cs"); _TAG = _os.environ.get("TAG", "")

ao=np.load("aif_slice21.npz")["ao"].astype(bool)
z=np.load("step2_slice21.npz"); mf=np.abs(z["mf"]).transpose(1,2,0).astype(np.float32); tmf=np.asarray(z["tmf"])
med=lambda v:np.median(v[ao],0); n0=lambda c:c/(np.median(c[:8])+1e-30)
Ld=lambda f:(np.abs(np.load(f"{RD}/{f}")).astype(np.float32) if os.path.exists(f"{RD}/{f}") else None)
tN=np.linspace(0,TA,342); mfc=n0(med(mf))
cfg=[("siren (w0=90)","outcoil_subspace_output_pw90_slice21.npy"),
     ("gabor / wire","outcoil_subspace_output_gabor_slice21.npy"),
     ("gaussian","outcoil_subspace_output_gaussian_slice21.npy")]
inp=Ld("outcoil_subspace_input_slice21.npy")
fig=plt.figure(figsize=(15,7)); gs=fig.add_gridspec(2,4,height_ratios=[1.1,1])
axc=fig.add_subplot(gs[0,:2])
axc.plot(tmf,mfc,color="0.45",lw=3,alpha=.85,label="model-free"); axc.plot(tN,n0(med(inp)),"k--",lw=1.3,label="input-coil")
for nm,f in cfg:
    v=Ld(f); axc.plot(tN,n0(med(v)),lw=1.3,label=nm) if v is not None else None
axc.set_title("aorta bolus, temporal activation (output-coil, rank16, w0=90)"); axc.set_xlabel("time (s)"); axc.legend(fontsize=8)
# peak ratio bars
axb=fig.add_subplot(gs[0,2:])
labs=["input"]+[c[0] for c in cfg]; vals=[n0(med(inp)).max()/mfc.max()]+[ (n0(med(Ld(f))).max()/mfc.max() if Ld(f) is not None else 0) for _,f in cfg]
axb.bar(range(len(labs)),vals,color=["k","C0","C1","C2"]); axb.axhline(1.0,ls=":",c="0.45")
axb.set_xticks(range(len(labs))); axb.set_xticklabels(labs,fontsize=8,rotation=15); axb.set_ylabel("aorta peak / model-free"); axb.set_title("bolus recovery"); axb.set_ylim(0,1.05)
# peak-frame images
ref=Ld(cfg[0][1]); m=ref.mean(-1)>ref.mean()*0.3; ys,xs=np.where(m); y0,y1,x0,x1=max(ys.min()-6,0),ys.max()+6,max(xs.min()-6,0),xs.max()+6
for j,(nm,f) in enumerate(cfg):
    axi=fig.add_subplot(gs[1,j]); v=Ld(f)
    if v is not None:
        pk=int(np.argmax(med(v))); im=v[y0:y1,x0:x1,pk]; axi.imshow(im,cmap="gray",vmin=0,vmax=np.percentile(im,99.5))
    axi.set_title(nm,fontsize=9); axi.axis("off")
axi=fig.add_subplot(gs[1,3]); pk=int(np.argmax(med(inp))); im=inp[y0:y1,x0:x1,pk]; axi.imshow(im,cmap="gray",vmin=0,vmax=np.percentile(im,99.5)); axi.set_title("input-coil (ref)",fontsize=9); axi.axis("off")
fig.suptitle("STEP 3: temporal activation comparison (real slice 21)",fontsize=12)
plt.tight_layout(); fig.savefig(f"{RD}/figures/step3_compare{_TAG}.png",dpi=115); print("peaks",dict(zip(labs,[round(v,2) for v in vals]))); print(f"SAVED figures/step3_compare{_TAG}.png DONE_S3")
