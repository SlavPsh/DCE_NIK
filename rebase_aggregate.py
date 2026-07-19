"""Unified re-baseline (render fix support=1.0): full-rank + factorized R vs CS.
held-out + swing + nav-corr + HF/CS + DISTS/HaarPSI vs CS-100."""
import re, sys, numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/nik-autoresearch/glue"); sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import eval as E, nik_adapter as NA
D = "/scratch/rnga/vvpshenov/DCE_NIK"; A = "/scratch/rnga/vvpshenov/presentation/assets"
jmap = dict(l.split() for l in open(f"{D}/rebase_jids.txt"))
freqmap = dict(l.split() for l in open(f"{D}/freq_jids.txt"))
cs100 = np.abs(np.load(f"{A}/arm1_cs100_sl13.npy")); cs70 = np.abs(np.load(f"{A}/arm1_cs70_sl13.npy")); csm = cs100.mean(-1)
roi = csm > np.quantile(csm, 0.55)
sh = NA.load_shared(f"/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"); sl = NA.load_slice(f"/scratch/rnga/vvpshenov/grasp_pro_py/results_ref", 13)
krad = np.asarray(sl["kdata_radial"]); vt = np.asarray(sh["view_time"]).ravel(); c0 = krad.shape[0]//2
dc = np.sqrt((np.abs(krad[c0]) ** 2).sum(-1)); o = np.argsort(vt); ts = vt[o]
dcs = np.clip(dc[o], np.percentile(dc,1), np.percentile(dc,99))
nav = np.convolve(np.pad(dcs,(30,30),mode="reflect"), np.ones(61)/61, mode="valid")[:len(dcs)]; nav/=np.median(nav[int(.35*len(nav)):])
def n01(a): a=np.asarray(a,float); return (a-a.min())/(a.max()-a.min()+1e-12)
def hf(img):
    im=img/(img.mean()+1e-12); Fp=np.abs(np.fft.fftshift(np.fft.fft2(im)))**2
    ny,nx=im.shape; y,x=np.indices((ny,nx)); r=np.hypot(y-ny//2,x-nx//2).astype(int)
    ps=np.bincount(r.ravel(),Fp.ravel())/np.maximum(np.bincount(r.ravel()),1); return ps[len(ps)//2:].sum()/ps.sum()
cshf=hf(csm)
def held(logglob):
    import glob
    for lg in glob.glob(logglob):
        for line in open(lg):
            m=re.search(r"restored best \(heldout ([\d.eE+-]+)\)", line)
            if m: return float(m.group(1))
    return float("nan")
def metrics(st, h):
    nt=st.shape[-1]; rr=st.mean(-1)>np.quantile(st.mean(-1),0.6)
    c=np.array([st[...,i][rr].mean() for i in range(nt)]); sw=(c.max()-c.min())/c.mean()*100
    tt=np.linspace(0,1,nt); nc=float(np.corrcoef(n01(c), np.interp(tt,ts,n01(nav)))[0,1])
    m=E.image_metrics(st.mean(-1), csm, roi)
    return h, sw, nc, hf(st.mean(-1))/cshf, m.get("DISTS",np.nan), m.get("HaarPSI",np.nan)
rows=[]
specs=[("full-rank", f"{D}/results_rebase_fullrank/nik_slice_13.npy", f"{D}/slurm-nik-rebase-{jmap.get('fullrank','X')}.out"),
       ("R=5",  f"{D}/results_rebase_r5/nik_slice_13.npy",  f"{D}/slurm-nik-rebase-{jmap.get('r5','X')}.out"),
       ("R=10", f"{D}/results_rebase_r10/nik_slice_13.npy", f"{D}/slurm-nik-rebase-{jmap.get('r10','X')}.out"),
       ("R=16", f"{D}/results_freq_base/nik_slice_13.npy",  f"{D}/slurm-nik-freq-{freqmap.get('base','X')}.out"),
       ("R=20", f"{D}/results_rebase_r20/nik_slice_13.npy", f"{D}/slurm-nik-rebase-{jmap.get('r20','X')}.out")]
print(f"{'model':10} {'held':>7} {'swing%':>7} {'navcorr':>8} {'HF/CS':>6} {'DISTS':>6} {'HaarPSI':>7}  (vs CS-100)")
for name, npy, lg in specs:
    try:
        st=np.abs(np.load(npy)); h,sw,nc,hfr,di,ha=metrics(st, held(lg))
        print(f"{name:10} {h:7.4f} {sw:7.1f} {nc:+8.3f} {hfr:6.2f} {di:6.3f} {ha:7.3f}")
    except FileNotFoundError: print(f"{name:10}   (pending)")
for name, arr in [("CS-70", cs70), ("CS-100", cs100)]:
    _,sw,nc,hfr,di,ha=metrics(arr, float("nan"))
    print(f"{name:10} {'-':>7} {sw:7.1f} {nc:+8.3f} {hfr:6.2f} {di:6.3f} {ha:7.3f}")
