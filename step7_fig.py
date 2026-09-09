"""fair spoke-matched grasp: nik (80% spokes) vs grasp f80-match (same 80% spokes) vs grasp f100 (100%). real slice 21."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RD="results/realdata_nik_vs_cs_figures"; TA=375.0
import os as _os
# reference-method plumbing. defaults = grasp-pro (unchanged). grasp v2:
#   CSD=/net/beegfs/users/P101440/grasp_v2/results_grasp_v2 CSPRE=gv2 TAG=_gv2
_CSD = _os.environ.get("CSD", "/net/beegfs/users/P101440/grasp_pro_py/results_spoke_cs")
_CSPRE = _os.environ.get("CSPRE", "cs"); _TAG = _os.environ.get("TAG", "")

ao=np.load("aif_slice21.npz")["ao"].astype(bool); kd=np.load("realkid_slice21.npz"); cx=kd["cortex"].astype(bool); md=kd["medulla"].astype(bool)
z=np.load("step2_slice21.npz"); mf=np.abs(z["mf"]).transpose(1,2,0).astype(np.float32); tmf=np.asarray(z["tmf"])
med=lambda v,m:np.median(v[m],0); n0=lambda c:c/(np.median(c[:8])+1e-30)
La=lambda p:(np.abs(np.load(p)).astype(np.float32) if os.path.exists(p) else None)
GR=_CSD
items=[("model-free",mf,tmf,dict(color="0.45",lw=3,alpha=.85)),
       ("nik input (80%)",La(f"{RD}/outcoil_subspace_input_slice21.npy"),np.linspace(0,TA,342),dict(c="C0",ls="--",lw=1.3)),
       ("nik output w0-90 (80%)",La(f"{RD}/outcoil_subspace_output_pw90_slice21.npy"),np.linspace(0,TA,342),dict(c="C2",lw=1.3)),
       ("grasp f100 (100%)",La(f"{GR}/{_CSPRE}_slice21_f100.npy"),np.linspace(0,TA,122),dict(c="k",ls=":",lw=1.6)),
       ("grasp f80-match (80%)",La(f"{GR}/{_CSPRE}_slice21_f80match.npy"),np.linspace(0,TA,122),dict(c="C3",ls=":",lw=1.6))]
fig,ax=plt.subplots(1,3,figsize=(15,4.3))
for j,(nm,mk) in enumerate([("aorta",ao),("cortex",cx),("medulla",md)]):
    for lab,v,tv,st in items:
        if v is not None: ax[j].plot(tv,n0(med(v,mk)),label=lab,zorder=2,**st)
    ax[j].set_title(nm); ax[j].set_xlabel("time (s)")
ax[0].legend(fontsize=7)
fig.suptitle("fair spoke-matched comparison: nik and grasp both on nik's 80% spokes (real slice 21)",fontsize=12)
plt.tight_layout(); fig.savefig(f"{RD}/figures/step7_fairgrasp{_TAG}.png",dpi=120); print(f"SAVED figures/step7_fairgrasp{_TAG}.png DONE_S7")
