import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, os, glob, finufft
import xph_pipeline as P, xph_common as X
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
_,_,_,_,nz,dims=P.build_train(dev); C=dims[3]
d=P.data(); tq=d["times"]; kx=d["kx"]; ky=d["ky"]; b1=d["b1"].astype(np.complex128); RO=b1.shape[0]; F=len(tq); tr=np.array(P.TRAIN_ANG)
b1c=np.ascontiguousarray(np.transpose(b1,(2,0,1))); rot=lambda im: np.roll(im[::-1,::-1],(1,1),axis=(0,1))
tag="sub16_w768_s0"; p=f"{P.OUT}/checkpoints/{tag}/ck_24000.pt"
m=P.make_model_g("wire_ff_subspace",768,P.FIX["k_sigma"],0,C,dev,rank=16,warmstart=False)
m.load_state_dict(torch.load(p,map_location=dev,weights_only=False)["state_dict"]); m.eval()
with torch.no_grad(): dyn=P.reconstruct_g(m,nz,tq,dev)
def misfit(imgs):
    num=denm=0.0
    for t in range(0,F,2):
        rr=np.abs(kx[t,tr]+1j*ky[t,tr]).reshape(-1)/0.5; low=rr<0.10
        fx=np.ascontiguousarray((2*np.pi*kx[t,tr]).reshape(-1)[low].astype(np.float64)); fy=np.ascontiguousarray((2*np.pi*ky[t,tr]).reshape(-1)[low].astype(np.float64))
        pred=finufft.nufft2d2(fx,fy,np.ascontiguousarray(imgs[:,:,t][None]*b1c),isign=-1,eps=1e-6)
        meas=d["kdata"][:,t,tr,:].reshape(C,-1)[:,low].astype(np.complex128)
        sc=np.sum(np.conj(pred)*meas)/(np.sum(np.abs(pred)**2)+1e-30); pred*=sc
        num+=(np.abs(pred-meas)**2).sum(); denm+=(np.abs(meas)**2).sum()
    return num/denm
dyn_r=np.stack([rot(dyn[:,:,t]) for t in range(F)],-1)
mn=misfit(dyn); mr=misfit(dyn_r)
print(f"forward(dyn NATIVE)  low-|k| misfit {mn:.3e} ({10*np.log10(1/mn):.1f} dB)")
print(f"forward(rot(dyn))    low-|k| misfit {mr:.3e} ({10*np.log10(1/mr):.1f} dB)")
print("DONE_RMF2")
