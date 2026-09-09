"""focused STEP-2 figure: w0 sweep + t_sigma arm, aorta bolus recovery vs input-coil + model-free."""
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
def C(fname):
    p=f"{RD}/{fname}"; v=np.abs(np.load(p)).astype(np.float32) if os.path.exists(p) else None
    return n0(med(v)) if v is not None else None
tN=np.linspace(0,TA,342)
mfc=n0(med(mf)); mfpk=mfc.max()
inp=C("outcoil_subspace_input_slice21.npy")
# explicit files: w0=30 baseline has NO tag; others tagged
W=[(30,"outcoil_subspace_output_slice21.npy"),(60,"outcoil_subspace_output_pw60_slice21.npy"),
   (90,"outcoil_subspace_output_pw90_slice21.npy"),(120,"outcoil_subspace_output_pw120_slice21.npy")]
TS=[(0.75,"outcoil_subspace_output_pw90ts075_slice21.npy"),(1.5,"outcoil_subspace_output_pw90_slice21.npy"),
    (3.0,"outcoil_subspace_output_pw90ts3_slice21.npy")]
fig,ax=plt.subplots(1,3,figsize=(15,4.2))
ax[0].plot(tmf,mfc,color="0.45",lw=3,alpha=.85,label="model-free"); ax[0].plot(tN,inp,"k--",lw=1.3,label="input-coil")
for w,f in W:
    c=C(f); ax[0].plot(tN,c,lw=1.2,label=f"w0={w}") if c is not None else None
ax[0].set_title("aorta, w0 sweep (rank16)"); ax[0].set_xlabel("time (s)"); ax[0].legend(fontsize=7)
ax[1].plot(tmf,mfc,color="0.45",lw=3,alpha=.85,label="model-free"); ax[1].plot(tN,inp,"k--",lw=1.3,label="input-coil")
for ts,f in TS:
    c=C(f); ax[1].plot(tN,c,lw=1.2,label=f"t_sigma={ts}") if c is not None else None
ax[1].set_title("aorta, t_sigma arm (w0=90)"); ax[1].set_xlabel("time (s)"); ax[1].legend(fontsize=7)
ws=[w for w,_ in W]; pks=[C(f).max()/mfpk for _,f in W]
ax[2].plot(ws,pks,"o-",lw=1.6,label="output-coil")
ax[2].axhline(inp.max()/mfpk,ls="--",c="k",label="input-coil"); ax[2].axhline(1.0,ls=":",c="0.45",label="model-free")
ax[2].set_title("aorta peak / model-free vs w0"); ax[2].set_xlabel("phi_w0"); ax[2].set_ylabel("peak ratio"); ax[2].legend(fontsize=7); ax[2].set_ylim(0,1.05)
fig.suptitle("STEP 2: w0 and t_sigma sweep, aorta bolus recovery (real slice 21, output-coil subspace)",fontsize=12)
plt.tight_layout(); fig.savefig(f"{RD}/figures/step2_compare{_TAG}.png",dpi=120); print(f"peaks w0={list(zip(ws,[round(p,2) for p in pks]))}"); print(f"SAVED figures/step2_compare{_TAG}.png DONE_S2")
