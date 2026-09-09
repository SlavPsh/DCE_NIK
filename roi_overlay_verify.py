"""ROI-placement verification: overlay ROI contours on the anatomy for real slice 21 + phantom slice 10.
Confirms ROIs sit on the intended structures (and shows the slice-13 aorta_roi.npy mistake)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B="/net/beegfs/users/P101440/DCE_NIK"; OUT=f"{B}/results/realdata_nik_vs_cs_figures/figures"
fig,ax=plt.subplots(1,2,figsize=(11,5.2))
# --- real slice 21 ---
z=np.load(f"{B}/step2_slice21.npz"); mf=np.abs(z["mf"]); anat=mf[np.argmin(abs(z["tmf"]-64))]  # peak frame
ao=np.asarray(np.load(f"{B}/aif_slice21.npz")["ao"]); wrong=np.load(f"{B}/aorta_roi.npy")
ax[0].imshow(anat,cmap="gray",vmax=np.percentile(anat[z["body"]],99))
ax[0].contour(ao,colors="lime",linewidths=1.2); ax[0].contour(wrong,colors="red",linewidths=1.0,linestyles="--")
ax[0].set_title("real slice 21 @64s\ngreen = AIF-gated aorta (correct) | red-dash = aorta_roi.npy (slice-13, WRONG)",fontsize=8); ax[0].axis("off")
# --- phantom slice 10 (label-based ROIs) ---
import xph_common as X; d=X.load_slice(10); tq=d["times"]; Tr=X.truth_at(10,tq); R=X.rois(10,d["labels"])
anatp=Tr[:,:,np.argmin(abs(tq-39))]
ax[1].imshow(anatp,cmap="gray",vmax=np.percentile(anatp[d["labels"]>0],99))
for nm,c in [("aorta","cyan"),("cortex","lime"),("medulla","orange")]: ax[1].contour(R[nm],colors=c,linewidths=1.2)
ax[1].set_title("phantom slice 10 @39s\ncyan=aorta(36) green=cortex(13) orange=medulla(37) [label-based]",fontsize=8); ax[1].axis("off")
fig.suptitle("ROI placement verification"); fig.tight_layout(); fig.savefig(f"{OUT}/fig6_roi_verification.png",dpi=130); plt.close(fig)
print("wrote fig6_roi_verification.png")
