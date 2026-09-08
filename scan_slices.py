import warnings; warnings.filterwarnings("ignore")
import numpy as np, xph_common as X
from scipy.ndimage import label as cclabel
print("slice | aorta central-comp: peak_s enh | kidney cortex/medulla vox")
for zi in [2,3,4,5,6,7]:
    d=X.load_slice(zi); lab=d["labels"]; tq=d["times"]; Tr=X.truth_at(zi,tq); N=lab.shape[0]
    ao=lab==36; cc,n=cclabel(ao)
    if n==0: print(f"  z{zi}: no aorta"); continue
    # central component
    bd=1e9; best=1
    for i in range(1,n+1):
        ys,xs=np.where(cc==i); dd=np.hypot(ys.mean()-N//2,xs.mean()-N//2)
        if dd<bd: bd=dd; best=i
    m=cc==best; c=np.median(Tr[m],0); b=c[:15].mean()
    cortex=int(((lab==23)|(lab==25)).sum()); medulla=int(((lab==24)|(lab==26)).sum())
    print(f"  z{zi}: aorta-central {int(m.sum()):3d}vox peak {tq[c.argmax()]:5.1f}s enh x{c.max()/(b+1e-6):.2f} | cortex {cortex} medulla {medulla}", flush=True)
