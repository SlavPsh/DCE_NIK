import warnings; warnings.filterwarnings("ignore")
import sys, os, numpy as np
sys.path.insert(0,"/scratch/rnga/vvpshenov/DCE_NIK")
import consolidated as C
B="/scratch/rnga/vvpshenov/DCE_NIK"; TA=375.0
ctx=C.slice_ctx(21); rois=ctx["rois"]; body=ctx["BODY"]
z=np.load(f"{B}/step2_slice21.npz"); mf=np.abs(z["mf"]).transpose(1,2,0).astype(np.float32); tmf=z["tmf"]
ROIS=[r for r in ("aorta","cortex","medulla") if rois.get(r) is not None and rois[r].sum()>0]
def bs(c): return c-np.median(c[:8])
mfc={r:bs(np.array([mf[...,i][rois[r]].mean() for i in range(mf.shape[-1])])) for r in ROIS}
def ft(n): e=np.linspace(0,TA,n+1); return 0.5*(e[:-1]+e[1:])
def rw(n):
    e=np.linspace(0,TA,n+1); o=np.zeros((mf.shape[0],mf.shape[1],n),np.float32)
    for g in range(n):
        m=(tmf>=e[g])&(tmf<e[g+1])
        if not m.any(): m=np.zeros_like(tmf,bool); m[np.argmin(np.abs(tmf-0.5*(e[g]+e[g+1])))]=True
        o[:,:,g]=mf[:,:,m].mean(2)
    return o
def fwhm(c,t):
    c=c-np.median(c[:8]); pk=c.max(); i=int(np.argmax(c)); h=pk/2; l=i; r=i
    while l>0 and c[l]>h: l-=1
    while r<len(c)-1 and c[r]>h: r+=1
    return t[r]-t[l]
print("phi_tv (Huber TV on temporal atoms), subspace R16, 1368 spokes + heldout. NIK-vs-NIK, so the mf ruler bias cancels.")
print(f"{'run':22} " + " ".join(f"{r+'C':>9}" for r in ROIS) + f" {'pk/ref':>7} {'FWHM s':>7}")
for w in ("0.0","0.01","0.1","1.0","10.0","100.0"):
    p=f"{B}/results_phitv_w{w}/nik_slice_21.npy"
    if not os.path.exists(p): continue
    v=np.abs(np.load(p)).astype(np.float32); R=rw(v.shape[-1]); v=v*(np.sum(v[body]*R[body])/(np.sum(v[body]**2)+1e-12)); t=ft(v.shape[-1])
    cur={r:np.interp(tmf,t,np.array([v[...,i][rois[r]].mean() for i in range(v.shape[-1])])) for r in ROIS}
    o=[np.linalg.norm(bs(cur[r])-mfc[r])/np.linalg.norm(mfc[r]) for r in ROIS]
    ca=np.array([v[...,i][rois["aorta"]].mean() for i in range(v.shape[-1])])
    print(f"{'w='+w:22} " + " ".join(f"{x:>9.4f}" for x in o) + f" {bs(cur['aorta']).max()/mfc['aorta'].max():>7.4f} {fwhm(ca,t):>7.1f}")
print("PHITV_SCORE_DONE")
