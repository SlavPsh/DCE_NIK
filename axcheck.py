import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np, inspect
sys.path.insert(0,"/scratch/rnga/vvpshenov/grasp_pro_py")
from fftc import fft2c_mri
print(inspect.getsource(fft2c_mri)[:300])
REF="/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
b1=np.asarray(np.load(f"{REF}/slice_21.npz")["b1"]); b1=b1/np.abs(b1).max()
K=fft2c_mri(b1); E=(np.abs(K)**2).sum(-1); c0=E.shape[0]//2
pk=np.unravel_index(np.argmax(E),E.shape)
print(f"b1 {b1.shape} -> K {K.shape}; peak {pk} centre {(c0,c0)}; r2 {100*E[c0-2:c0+3,c0-2:c0+3].sum()/E.sum():.2f}% r8 {100*E[c0-8:c0+9,c0-8:c0+9].sum()/E.sum():.2f}%")
xx,yy=np.meshgrid(np.arange(384)-192,np.arange(384)-192,indexing="ij")
g=np.exp(-(xx**2+yy**2)/(2*80.0**2)).astype(np.complex64)[:,:,None]
Kg=fft2c_mri(g); Eg=(np.abs(Kg)**2).sum(-1)
pkg=np.unravel_index(np.argmax(Eg),Eg.shape)
print(f"CONTROL smooth gaussian: peak {pkg}; r2 {100*Eg[190:195,190:195].sum()/Eg.sum():.2f}% (expect ~100)")
