"""does the trained model fit the TRAINING data at every |k| bin? same setup as
_diag_radial_sharpness.py through norm.fit, then predict on kc_tr and bin
residual by |k|. R^2 per bin tells us:
  high R^2 across ALL |k|   -> training succeeded; peripheral image blur is
                                eval-side (coord-grid mismatch / FFT-side).
  R^2 drops with |k|        -> training itself fails at periphery; bandwidth /
                                density / lr is the real bottleneck."""
import numpy as np, torch
from nik_io import load_event
from nik_train import prepare_tensors
from nik_recon import ifft1d_kz_to_z, make_multicoil_time_radial_dataset
from kspace_normalization import compute_radius, KSpaceNormalizer
from nik_model import WIRE_KXY_COIL_T_REIM

dev="cpu"; seed=0; sub=0.7
H,D,W0,S0,CE = 384,8,62.0,15.0,8
fp='/scratch/rnga/vvpshenov/XCAT-ERIC/results/simulation_results_20260527T175428.mat'
ckpt='runs/20260529_161643_lyrical-amaranth-kiwi_mct_carteval/model_best.pth'  # p0 dcf-off
NB=12

torch.manual_seed(seed); np.random.seed(seed)
ev=load_event(fp, load_images=False, load_coil_maps=False)
k_np=np.transpose(ev['k'],(0,2,1,3)); traj_np=np.transpose(ev['traj'],(0,2,1,3))
T,S,C,RO=k_np.shape
k_t,traj_t,scales,dims,_=prepare_tensors(k_np,traj_np,data_device=dev)
k_img,n_z,n_ro,_=ifft1d_kz_to_z(k_t,traj_t,t_frame=0); z=n_z//2
st=np.asarray(ev['spoke_timing_dce'])
if st.shape[0]==T and st.shape[1]!=T: st=st.T
tmax=float(st.max())
x_all,t_all,coil_all,y_all_raw,spoke_id,_,_,meta=make_multicoil_time_radial_dataset(
    k_img,traj_t,scales,dims,z_slice_idx=z,n_slices=n_z,compute_device=dev,
    spoke_timing_dce=st,t_max_ms=tmax)
nuq=int(spoke_id.max())+1; ntr=max(1,int(nuq*sub))
g=torch.Generator().manual_seed(seed); perm=torch.randperm(nuq,generator=g)
train_idx=torch.where(torch.isin(spoke_id,perm[:ntr]))[0]

kc=x_all[train_idx]; tc=t_all[train_idx]; cc=coil_all[train_idx]; y_raw=y_all_raw[train_idx]
norm=KSpaceNormalizer()
norm.fit(kc,y_raw,dcf=torch.ones(train_idx.numel()),envelope_bins=128,
         envelope_statistic='weighted_rms',envelope_smooth_method='moving_average',
         envelope_smooth_width=5,envelope_floor_fraction=1e-3,global_scale_method='weighted_rms')
y_norm=norm.normalize(kc,y_raw)              # what the model is trained to output (re/im)
r=compute_radius(kc).numpy()
rmax=r.max()
print(f'training points={kc.shape[0]}, r_max={rmax:.4f}')

model=WIRE_KXY_COIL_T_REIM(n_coils=C,coil_embed_dim=CE,hidden=H,depth=D,w0=W0,s0=S0)
model.load_state_dict(torch.load(ckpt,map_location=dev)['model_state_dict']); model.eval()

# predict in chunks (CPU)
N=kc.shape[0]; CHUNK=131072
preds=np.empty((N,2),dtype=np.float32)
with torch.no_grad():
    for i in range(0,N,CHUNK):
        j=min(N,i+CHUNK)
        preds[i:j]=model(kc[i:j], tc[i:j], cc[i:j]).cpu().numpy()
        if (i//CHUNK)%20==0: print(f'  predicted {j}/{N}')

y_np=y_norm.numpy().astype(np.float32)
res=preds-y_np                                # (N,2) residual in normalized space
res_mag2=(res**2).sum(axis=1)                 # |residual|^2
sig_mag2=(y_np**2).sum(axis=1)                # |signal|^2

# bin by |k|
edges=np.linspace(0, rmax, NB+1)
bi=np.clip(np.digitize(r, edges)-1, 0, NB-1)
res_per=np.bincount(bi, res_mag2, minlength=NB)
sig_per=np.bincount(bi, sig_mag2, minlength=NB)
cnt_per=np.bincount(bi, minlength=NB)
mean_res=res_per/np.maximum(cnt_per,1)
mean_sig=sig_per/np.maximum(cnt_per,1)
R2=1.0 - res_per/np.maximum(sig_per,1e-12)    # fraction of variance explained
nmse=res_per/np.maximum(sig_per,1e-12)
rfrac=(edges[:-1]+edges[1:])/2/rmax

print('\n|k|_frac | count       | mean_sig^2 | mean_res^2 | NMSE   | R^2')
for i in range(NB):
    print(f'  {rfrac[i]:.2f}   | {cnt_per[i]:11d} | {mean_sig[i]:10.4f} | {mean_res[i]:10.4f} | {nmse[i]:.4f} | {R2[i]:+.3f}')
print(f'\nglobal R^2 = {1.0 - res_per.sum()/sig_per.sum():+.3f}')
def band(arr, lo, hi):
    m=(rfrac>=lo)&(rfrac<hi); return float(arr[m].mean()) if m.any() else float('nan')
print(f'R^2  inner(0-33%)={band(R2,0,.33):+.3f}  mid(33-66%)={band(R2,.33,.66):+.3f}  outer(66-100%)={band(R2,.66,1.01):+.3f}')
print('\nverdict:')
print('  flat-high R^2 across |k|  -> training fit OK at periphery; image-edge blur is EVAL-side (coord/FFT).')
print('  falling R^2 with |k|      -> training fails at high |k|; bandwidth / lr / density is the real fix.')
