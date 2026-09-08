"""L2c: is the grid-query round-trip loss a fixable COARSE-GRID aliasing bug, or INHERENT network
non-self-consistency? Render sub16_s0 at k-grid oversampling factor 1x and 2x (2x = denser grid over
the same [-1,1] extent -> 2x FOV image -> crop central RO), forward-project to spokes, compare S1 dB.
If 2x recovers most of the ~18 dB -> coarse-grid aliasing (fixable). If ~unchanged -> inherent."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob, finufft
import xph_pipeline as P, nik_adapter as A
from fftc import ifft2c_mri, crop_img
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
_,_,_,_,nz,dims=P.build_train(dev); C=dims[3]
d=P.data(); tq=d["times"]; kx=d["kx"]; ky=d["ky"]; RO=d["b1"].shape[0]; F=len(tq); tr=np.array(P.TRAIN_ANG); Ttime=float(tq.max())
rot=lambda im: np.roll(im[::-1,::-1],(1,1),axis=(0,1))
m=P.make_model_g("wire_ff_subspace",768,P.FIX["k_sigma"],0,C,dev,rank=16,warmstart=False)
p=f"{P.OUT}/checkpoints/sub16_w768_s0/ck_24000.pt"; m.load_state_dict(torch.load(p,map_location=dev,weights_only=False)["state_dict"]); m.eval()
tn_all=torch.tensor([2*tq[t]/Ttime-1 for t in range(F)],dtype=torch.float32,device=dev)
with torch.no_grad(): Phi=m.basis(tn_all).cpu().numpy()[:,:,0].astype(np.complex128); Rr=Phi.shape[1]
FR=np.arange(0,F,4)
def cart_at(N):  # network coeff k-space on an NxN grid over [-1,1]
    g=torch.from_numpy(A.cartesian_grid(N)).to(dev); out=np.zeros((N,N,Rr,C),np.complex128)
    with torch.no_grad():
        for c in range(C):
            Amp=m.amplitudes(g,torch.full((g.shape[0],),c,dtype=torch.long,device=dev))
            o=np.zeros((g.shape[0],Rr),np.complex128)
            for r in range(Rr): pr=nz.denormalize(g,Amp[:,r,:].contiguous()); o[:,r]=(pr[:,0]+1j*pr[:,1]).cpu().numpy()
            out[:,:,:,c]=o.reshape(N,N,Rr)
    return out
cart1=cart_at(RO); cart2=cart_at(2*RO)
db=lambda n,dd:10*np.log10((dd+1e-30)/(n+1e-30))
def measure(cart,N,ov):
    num=de=0.0
    for t in FR:
        fx=np.ascontiguousarray((2*np.pi*kx[t,tr]).reshape(-1).astype(np.float64)); fy=np.ascontiguousarray((2*np.pi*ky[t,tr]).reshape(-1).astype(np.float64))
        cx=torch.tensor(np.stack([2*kx[t,tr].reshape(-1),2*ky[t,tr].reshape(-1)],1),dtype=torch.float32,device=dev)
        tt=torch.full((cx.shape[0],),2*tq[t]/Ttime-1,dtype=torch.float32,device=dev)
        ref=np.empty((C,cx.shape[0]),np.complex128)
        with torch.no_grad():
            for c in range(C):
                pr=nz.denormalize(cx,m(cx,tt,torch.full((cx.shape[0],),c,dtype=torch.long,device=dev))); ref[c]=(pr[:,0]+1j*pr[:,1]).cpu().numpy()
        S=[]
        for c in range(C):
            img=ifft2c_mri(cart[:,:,:,c]@Phi[t])            # NxN image
            if ov>1: img=crop_img(img,RO,RO)*ov             # crop central FOV; *ov keeps ortho scale
            S.append(finufft.nufft2d2(fx,fy,np.ascontiguousarray(rot(img)),isign=-1,eps=1e-6))
        S=np.stack(S); sc=np.sum(np.conj(S)*ref)/(np.sum(np.abs(S)**2)+1e-30); S=sc*S
        num+=(np.abs(S-ref)**2).sum(); de+=(np.abs(ref)**2).sum()
    return db(num,de)
print(f"S1 grid-query, 1x grid (RO={RO})   : {measure(cart1,RO,1):.2f} dB")
print(f"S1 grid-query, 2x oversampled+crop : {measure(cart2,2*RO,2):.2f} dB")
print("-> 2x recovers most of the ~18 dB => coarse-grid aliasing (FIXABLE); ~unchanged => INHERENT")
print("DONE_L2C")
