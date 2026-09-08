"""Multi-metric cross-check for the P0/P3 batch: does the aorta noise-win survive on
held-out / swing / nav-corr / perceptual? held-out parsed from logs (verified ordering)."""
import re, sys, numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/nik-autoresearch/glue"); sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import eval as E, nik_adapter as NA
D = "/scratch/rnga/vvpshenov/DCE_NIK"; A = "/scratch/rnga/vvpshenov/presentation/assets"
GP = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"
cs100 = np.abs(np.load(f"{A}/arm1_cs100_sl13.npy")); csm = cs100.mean(-1)
cs70p = f"{A}/arm1_cs70_sl13.npy"; cs70 = np.abs(np.load(cs70p)) if __import__("os").path.exists(cs70p) else None
roi = csm > np.quantile(csm, 0.55)
aroi = np.load(f"{D}/aorta_roi.npy")
sh = NA.load_shared(GP); sl = NA.load_slice(GP, 13)
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
def metrics(st, h):
    nt=st.shape[-1]; rr=st.mean(-1)>np.quantile(st.mean(-1),0.6)
    c=np.array([st[...,i][rr].mean() for i in range(nt)]); sw=(c.max()-c.min())/c.mean()*100      # global swing
    ca=np.array([st[...,i][aroi].mean() for i in range(nt)]); swa=(ca.max()-ca.min())/ca.mean()*100  # aorta swing
    tt=np.linspace(0,1,nt); nc=float(np.corrcoef(n01(c), np.interp(tt,ts,n01(nav)))[0,1])
    m=E.image_metrics(st.mean(-1), csm, roi)
    return h, sw, swa, nc, m.get("DISTS",np.nan), m.get("HaarPSI",np.nan)
# label -> (jobid for held-out, results dir)
J = dict(base="2972531", bl_ts1p0="2972532", bl_ts0p7="2972533", tv0p03="2972534",
         tv0p1="2972535", pk_soft="2972536", pk_hard="2972537")
NAME = dict(base="base t_sig1.5", bl_ts1p0="bandlimit 1.0", bl_ts0p7="bandlimit 0.7",
            tv0p03="TV w0.03", tv0p1="TV w0.1", pk_soft="PK soft", pk_hard="PK hard")
def held(j):
    for line in open(f"{D}/logs/slurm-nik-pktv-{j}.out"):
        m=re.search(r"restored best \(heldout ([\d.eE+-]+)\)", line)
        if m: return float(m.group(1))
    return float("nan")
print(f"{'run':16} {'held':>7} {'swing%':>7} {'aortaSw%':>8} {'navcorr':>8} {'DISTS':>6} {'HaarPSI':>7}  (vs CS-100; held lower=fit better)")
for lab in ["base","bl_ts1p0","bl_ts0p7","tv0p03","tv0p1","pk_soft","pk_hard"]:
    st=np.abs(np.load(f"{D}/results_pktv_{lab}/nik_slice_13.npy"))
    h,sw,swa,nc,di,ha=metrics(st, held(J[lab]))
    print(f"{NAME[lab]:16} {h:7.4f} {sw:7.1f} {swa:8.1f} {nc:+8.3f} {di:6.3f} {ha:7.3f}")
for name, arr in [("CS-70", cs70), ("CS-100", cs100)]:
    if arr is None: continue
    _,sw,swa,nc,di,ha=metrics(arr, float("nan"))
    print(f"{name:16} {'-':>7} {sw:7.1f} {swa:8.1f} {nc:+8.3f} {di:6.3f} {ha:7.3f}")
