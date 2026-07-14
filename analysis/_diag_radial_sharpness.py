"""memory-light: reconstruct ONE frame from the trained model and measure
sharpness (local gradient energy) as a function of IMAGE radius. settles whether
the model blur is uniform (case A) or grows toward the image edge (case B).
ratio model/gt and cs/gt per annulus: flat -> uniform, falling -> radius-dependent."""
import sys, numpy as np, torch, h5py
from nik_io import load_event, _load_slice_profile, _bin_gt_to_dce
from nik_train import prepare_tensors
from nik_recon import ifft1d_kz_to_z, make_multicoil_time_radial_dataset
from kspace_normalization import compute_radius, KSpaceNormalizer
from nik_model import WIRE_KXY_COIL_T_REIM

dev="cpu"; seed=0; sub=0.7; tpick=90
H,D,W0,S0,CE = 384,8,62.0,15.0,8
fp='/scratch/rnga/vvpshenov/XCAT-ERIC/results/simulation_results_20260527T175428.mat'
default_ckpt='runs/20260529_161643_lyrical-amaranth-kiwi_mct_carteval/model_best.pth'  # p0 dcf-off baseline
ckpt = sys.argv[1] if len(sys.argv) > 1 else default_ckpt
print(f'ckpt = {ckpt}')

torch.manual_seed(seed); np.random.seed(seed)
ev=load_event(fp, load_images=True, load_coil_maps=True)
k_np=np.transpose(ev['k'],(0,2,1,3)); traj_np=np.transpose(ev['traj'],(0,2,1,3))
T,S,C,RO=k_np.shape
k_t,traj_t,scales,dims,_=prepare_tensors(k_np,traj_np,data_device=dev)
k_img,n_z,n_ro,_=ifft1d_kz_to_z(k_t,traj_t,t_frame=0); z=n_z//2
sx,sy,_=scales; sx=float(sx); sy=float(sy)

st=np.asarray(ev['spoke_timing_dce']);
if st.shape[0]==T and st.shape[1]!=T: st=st.T
tmax=float(st.max())
x_all,t_all,coil_all,y_all_raw,spoke_id,_,_,meta=make_multicoil_time_radial_dataset(
    k_img,traj_t,scales,dims,z_slice_idx=z,n_slices=n_z,compute_device=dev,
    spoke_timing_dce=st,t_max_ms=tmax)
nuq=int(spoke_id.max())+1; ntr=max(1,int(nuq*sub))
g=torch.Generator().manual_seed(seed); perm=torch.randperm(nuq,generator=g)
train_idx=torch.where(torch.isin(spoke_id,perm[:ntr]))[0]
kc_tr=x_all[train_idx]; y_tr=y_all_raw[train_idx]
norm=KSpaceNormalizer()
norm.fit(kc_tr,y_tr,dcf=torch.ones(train_idx.numel()),envelope_bins=128,
         envelope_statistic='weighted_rms',envelope_smooth_method='moving_average',
         envelope_smooth_width=5,envelope_floor_fraction=1e-3,global_scale_method='weighted_rms')
rmax=float(compute_radius(kc_tr).max())

# --- ONE-frame cart: build coil/gt grid manually (no full synthesis) ---
gt_img=ev['gt_img']; coil_maps=ev['coil_maps']; sp=_load_slice_profile(fp, gt_img.shape[1])
nC,nkz,RLp,APp=coil_maps.shape
gt_binned=_bin_gt_to_dce(gt_img, ev.get('gt_tim'), ev.get('rc_tim'), T)   # (T,kz,RL,AP)
nRL,nAP=gt_binned.shape[2],gt_binned.shape[3]
pRL=(RLp-nRL)//2; pAP=(APp-nAP)//2
gt_pad_f=np.pad(gt_binned[tpick],((0,0),(pRL,pRL),(pAP,pAP)))   # (kz,RLp,APp)

kx=torch.fft.fftshift(torch.fft.fftfreq(RLp)); ky=torch.fft.fftshift(torch.fft.fftfreq(APp))
KY,KX=torch.meshgrid(ky,kx,indexing='ij')
nky,nkx=APp,RLp
xc=torch.stack([KX.reshape(-1)/sx, KY.reshape(-1)/sy],dim=1).float()
Npt=xc.shape[0]
tnorm=float((ev['rc_tim'].reshape(-1)[tpick]*1000.0/tmax)*2-1)
tcol=torch.full((Npt,),tnorm)
cr=compute_radius(xc); disk=(cr<=rmax+1e-6).view(nky,nkx).numpy()

model=WIRE_KXY_COIL_T_REIM(n_coils=C,coil_embed_dim=CE,hidden=H,depth=D,w0=W0,s0=S0)
model.load_state_dict(torch.load(ckpt,map_location=dev)['model_state_dict']); model.eval()

sens=coil_maps[:,z].astype(np.complex64); denom=np.sum(np.abs(sens)**2,axis=0)+1e-10
pred_coil=[]
with torch.no_grad():
    for c in range(C):
        ci=torch.full((Npt,),c,dtype=torch.long)
        pn=model(xc,tcol,ci); pdn=norm.denormalize(xc,pn).numpy()
        kp=((pdn[:,0]+1j*pdn[:,1]).reshape(nky,nkx))*disk
        pred_coil.append(np.fft.fftshift(np.fft.ifft2(kp)).T)
pred_coil=np.stack(pred_coil,axis=0)
img_model=np.abs(np.sum(np.conj(sens)*pred_coil,axis=0)/denom)

with h5py.File(fp,'r') as h:
    cs_img=np.abs(h['/results/images/Recon/img'][tpick,z])
natRL,natAP=cs_img.shape
r0=(RLp-natRL)//2; a0=(APp-natAP)//2
crop=lambda im:im[r0:r0+natRL,a0:a0+natAP]
gt_n=np.abs(crop(gt_pad_f[z])); mdl=crop(img_model)
n=lambda a:a/(a.max()+1e-12); gt_n,mdl,cs_img=n(gt_n),n(mdl),n(cs_img)

# --- radial sharpness profile: local gradient energy vs IMAGE radius ---
def grad_energy(img):
    gy,gx=np.gradient(img.astype(np.float64))
    return gy**2+gx**2   # per-pixel sharpness

def radial_profile(vals, nbins=14):
    Hh,Ww=vals.shape; cy,cx=Hh/2.0,Ww/2.0
    Y,X=np.indices((Hh,Ww))
    r=np.sqrt((Y-cy)**2+(X-cx)**2)
    rmaxv=r.max(); ri=np.clip((r/rmaxv*nbins).astype(int),0,nbins-1)
    out=np.array([vals[ri==b].mean() for b in range(nbins)])
    edges=np.array([(b+0.5)/nbins for b in range(nbins)])  # fractional radius
    return edges, out

eg_gt=grad_energy(gt_n); eg_md=grad_energy(mdl); eg_cs=grad_energy(cs_img)
rr,pg=radial_profile(eg_gt); _,pm=radial_profile(eg_md); _,pc=radial_profile(eg_cs)
ratio_m=pm/(pg+1e-12); ratio_c=pc/(pg+1e-12)

print('frame',tpick,'z',z,'ckpt p0 (dcf off)')
print('radius_frac | grad-energy: GT / CS / MODEL | sharpness ratio: CS/GT  MODEL/GT')
for i in range(len(rr)):
    print('  %.2f      | %9.3e %9.3e %9.3e |   %.3f    %.3f'%(
        rr[i], pg[i], pc[i], pm[i], ratio_c[i], ratio_m[i]))
print()
# summarize trend: ratio in inner third vs outer third
def band(arr,lo,hi):
    i0=int(len(arr)*lo); i1=int(len(arr)*hi); return float(np.mean(arr[i0:i1]))
print('MODEL/GT sharpness  inner(0-33%%)=%.3f  mid(33-66%%)=%.3f  outer(66-100%%)=%.3f'%(
    band(ratio_m,0,.33), band(ratio_m,.33,.66), band(ratio_m,.66,1.0)))
print('CS/GT    sharpness  inner(0-33%%)=%.3f  mid(33-66%%)=%.3f  outer(66-100%%)=%.3f'%(
    band(ratio_c,0,.33), band(ratio_c,.33,.66), band(ratio_c,.66,1.0)))
print('=> falling MODEL/GT across inner->outer confirms radius-dependent (case B) blur.')
