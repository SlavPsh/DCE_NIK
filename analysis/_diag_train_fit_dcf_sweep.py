"""run the training-fit-vs-|k| diagnostic across all 4 dcf_power checkpoints.
data prep + norm.fit done ONCE; only the model state is swapped per checkpoint.
verdict: if dcf_power increases the outer-third R^2, the loss-reweighting axis
is the right one. if all 4 stay flat-bad at high |k|, dcf_power is not the fix."""
import numpy as np, torch
from nik_io import load_event
from nik_train import prepare_tensors
from nik_recon import ifft1d_kz_to_z, make_multicoil_time_radial_dataset
from kspace_normalization import compute_radius, KSpaceNormalizer
from nik_model import WIRE_KXY_COIL_T_REIM

dev="cpu"; seed=0; sub=0.7
H,D,W0,S0,CE = 384,8,62.0,15.0,8
fp='/scratch/rnga/vvpshenov/XCAT-ERIC/results/simulation_results_20260527T175428.mat'
NB=12
RUNS=[
 ("p0    (dcf off)",       "runs/20260529_161643_lyrical-amaranth-kiwi_mct_carteval/model_best.pth"),
 ("p0p3  (dcf on, 0.3)",   "runs/20260529_162007_literate-attractive-dodo_mct_carteval/model_best.pth"),
 ("p0p5  (dcf on, 0.5)",   "runs/20260529_164030_wooden-impossible-jellyfish_mct_carteval/model_best.pth"),
 ("p1p0  (dcf on, 1.0)",   "runs/20260529_164339_dramatic-rare-numbat_mct_carteval/model_best.pth"),
]

torch.manual_seed(seed); np.random.seed(seed)
print("loading data + building dataset (once) ...")
ev=load_event(fp, load_images=False, load_coil_maps=False)
k_np=np.transpose(ev['k'],(0,2,1,3)); traj_np=np.transpose(ev['traj'],(0,2,1,3))
T,S,C,RO=k_np.shape
k_t,traj_t,scales,dims,_=prepare_tensors(k_np,traj_np,data_device=dev)
k_img,n_z,n_ro,_=ifft1d_kz_to_z(k_t,traj_t,t_frame=0); z=n_z//2
st=np.asarray(ev['spoke_timing_dce'])
if st.shape[0]==T and st.shape[1]!=T: st=st.T
tmax=float(st.max())
x_all,t_all,coil_all,y_all_raw,spoke_id,_,_,_=make_multicoil_time_radial_dataset(
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
y_norm=norm.normalize(kc,y_raw).numpy().astype(np.float32)
r=compute_radius(kc).numpy(); rmax=r.max()
print(f'  training points={kc.shape[0]}, r_max={rmax:.4f}')

edges=np.linspace(0, rmax, NB+1); bi=np.clip(np.digitize(r,edges)-1,0,NB-1)
rfrac=(edges[:-1]+edges[1:])/2/rmax

def predict_and_R2(ckpt):
    model=WIRE_KXY_COIL_T_REIM(n_coils=C,coil_embed_dim=CE,hidden=H,depth=D,w0=W0,s0=S0)
    model.load_state_dict(torch.load(ckpt,map_location=dev)['model_state_dict']); model.eval()
    N=kc.shape[0]; CHUNK=131072
    preds=np.empty((N,2),dtype=np.float32)
    with torch.no_grad():
        for i in range(0,N,CHUNK):
            j=min(N,i+CHUNK)
            preds[i:j]=model(kc[i:j], tc[i:j], cc[i:j]).cpu().numpy()
    res=preds-y_norm
    res_mag2=(res**2).sum(axis=1); sig_mag2=(y_norm**2).sum(axis=1)
    res_per=np.bincount(bi, res_mag2, minlength=NB)
    sig_per=np.bincount(bi, sig_mag2, minlength=NB)
    R2_per=1.0 - res_per/np.maximum(sig_per,1e-12)
    glob = 1.0 - res_per.sum()/sig_per.sum()
    def band(arr, lo, hi):
        m=(rfrac>=lo)&(rfrac<hi); return float(arr[m].mean()) if m.any() else float('nan')
    return R2_per, glob, band(R2_per,0,.33), band(R2_per,.33,.66), band(R2_per,.66,1.01)

results=[]
for lbl,ckpt in RUNS:
    print(f"\npredicting: {lbl}  ({ckpt})")
    R2p,g,binn,bmid,bout = predict_and_R2(ckpt)
    results.append((lbl,R2p,g,binn,bmid,bout))
    print(f"  R^2 inner={binn:+.3f}  mid={bmid:+.3f}  outer={bout:+.3f}  global={g:+.3f}")

print("\n================ R^2 vs |k| (per dcf_power) ================")
hdr="|k|_frac  " + "".join(f"| {lbl:<22s}" for lbl,_,_,_,_,_ in results)
print(hdr)
for i in range(NB):
    row=f"  {rfrac[i]:.2f}    "
    for _,R2p,*_ in results:
        row += f"|       {R2p[i]:+.3f}         "
    print(row)
print("\n=== summary bands ===")
print(f"{'label':<25s}  {'inner':>7s}  {'mid':>7s}  {'outer':>7s}  {'global':>7s}")
for lbl,_,g,binn,bmid,bout in results:
    print(f"{lbl:<25s}  {binn:+.3f}  {bmid:+.3f}  {bout:+.3f}  {g:+.3f}")
print("\n=> outer R^2 RISING with dcf_power -> reweighting is the right axis.")
print("=> outer R^2 STAYS FLAT at ~0 -> dcf_power does not help; need representational fix.")
