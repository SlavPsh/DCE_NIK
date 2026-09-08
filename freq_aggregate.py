"""Aggregate the freq/depth sweep (ranked model, render fix support=1.0). Per config:
held-out + swing + HF-ratio + perceptual (SSIM/DISTS/HaarPSI) vs CS-100. Now that the render
is fixed, perceptual-vs-CS is the real spatial comparison."""
import re, sys, numpy as np
sys.path.insert(0, "/scratch/rnga/vvpshenov/nik-autoresearch/glue")
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py")
import eval as E, nik_adapter as NA
D = "/scratch/rnga/vvpshenov/DCE_NIK"; A = "/scratch/rnga/vvpshenov/presentation/assets"
jmap = dict(l.split() for l in open(f"{D}/freq_jids.txt"))
cs = np.abs(np.load(f"{A}/arm1_cs100_sl13.npy")); csm = cs.mean(-1)
roi = csm > np.quantile(csm, 0.55)
sh = NA.load_shared("/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"); sl = NA.load_slice("/scratch/rnga/vvpshenov/grasp_pro_py/results_ref", 13)
krad = np.asarray(sl["kdata_radial"]); vt = np.asarray(sh["view_time"]).ravel(); c0 = krad.shape[0]//2
dc = np.sqrt((np.abs(krad[c0]) ** 2).sum(-1)); o = np.argsort(vt); ts = vt[o]
dcs = np.clip(dc[o], np.percentile(dc,1), np.percentile(dc,99))
nav = np.convolve(np.pad(dcs,(30,30),mode="reflect"), np.ones(61)/61, mode="valid")[:len(dcs)]; nav/=np.median(nav[int(.35*len(nav)):])
def n01(a): a=np.asarray(a,float); return (a-a.min())/(a.max()-a.min()+1e-12)
def hf(img):
    im=img/(img.mean()+1e-12); Fp=np.abs(np.fft.fftshift(np.fft.fft2(im)))**2
    ny,nx=im.shape; y,x=np.indices((ny,nx)); r=np.hypot(y-ny//2,x-nx//2).astype(int)
    ps=np.bincount(r.ravel(),Fp.ravel())/np.maximum(np.bincount(r.ravel()),1); return ps[len(ps)//2:].sum()/ps.sum()
cshf = hf(csm)
def heldout(tag):
    try:
        for line in open(f"{D}/logs/slurm-nik-freq-{jmap[tag]}.out"):
            m=re.search(r"restored best \(heldout ([\d.eE+-]+)\)", line)
            if m: return float(m.group(1))
    except Exception: pass
    return float("nan")
print(f"{'config':8} {'held':>7} {'swing%':>7} {'HF/CS':>6} {'SSIM':>6} {'DISTS':>6} {'HaarPSI':>7}   (perc vs CS-100)")
for tag in [l.split()[0] for l in open(f"{D}/freq_jids.txt")]:
    try:
        rec=np.abs(np.load(f"{D}/results_freq_{tag}/nik_slice_13.npy")); st=rec.mean(-1); nt=rec.shape[-1]
        rr=st>np.quantile(st,0.6); cc=np.array([rec[...,i][rr].mean() for i in range(nt)]); sw=(cc.max()-cc.min())/cc.mean()*100
        acc={"ssim":[],"DISTS":[],"HaarPSI":[]}
        for i in range(0,nt,max(1,nt//12)):
            m=E.image_metrics(st, csm, roi)  # static-vs-static (cheap, representative)
        m=E.image_metrics(st, csm, roi)
        print(f"{tag:8} {heldout(tag):7.4f} {sw:7.1f} {hf(st)/cshf:6.2f} {m.get('ssim',np.nan):6.3f} {m.get('DISTS',np.nan):6.3f} {m.get('HaarPSI',np.nan):7.3f}")
    except FileNotFoundError:
        print(f"{tag:8} {heldout(tag):7.4f}   (recon pending)")
print(f"\nCS-100 self HF-ratio {cshf:.3g} (HF/CS=1.0 means matches CS high-freq)")
