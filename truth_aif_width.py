"""independent physiological anchor: the XCAT ground-truth AIF, never reconstructed.
also the analytic Cosine4AIF used to generate it, if present."""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np
sys.path.insert(0,"/scratch/rnga/vvpshenov/DCE_NIK")
import xph_pipeline as P, xph_common as X
d=P.data(); tq=d["times"]; Rz=X.rois(P.ZI,d["labels"]); Tr=X.truth_at(P.ZI,tq)
def width(c,t):
    c=c-np.median(c[:8]); pk=c.max(); i=int(np.argmax(c)); h=pk/2
    l=i
    while l>0 and c[l]>h: l-=1
    r=i
    while r<len(c)-1 and c[r]>h: r+=1
    return t[r]-t[l], t[i], pk
c=Tr[Rz["aorta"]].mean(0)
w,ttp,pk=width(c,tq)
print(f"XCAT GROUND-TRUTH aorta AIF (never reconstructed):")
print(f"  FWHM {w:.1f} s   ttp {ttp:.1f} s   peak {pk:.4f}")
import os
p=f"{P.OUT.rsplit('/',2)[0]}/aif_slice21.npz"
if os.path.exists(f"/scratch/rnga/vvpshenov/DCE_NIK/aif_slice21.npz"):
    z=np.load("/scratch/rnga/vvpshenov/DCE_NIK/aif_slice21.npz")
    print(f"  (in-vivo aif_slice21.npz keys: {list(z.files)})")
