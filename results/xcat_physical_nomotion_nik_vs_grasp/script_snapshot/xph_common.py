"""Physical no-motion XCAT -> per-slice 2D radial, shared by NIK and GRASP-Pro (identical measurements).
Aligned stack-of-stars: z-FFT across the 16 partitions is valid. Truth = GroundTruth SPGR signal
(coil-combined magnitude) sampled at query times. XCAT is the only accuracy reference."""
import warnings; warnings.filterwarnings("ignore")
import h5py, numpy as np
SIM = "/scratch/rnga/vvpshenov/XCAT-ERIC/results/simulation_results_20260816T210718.mat"  # no-motion (respPeriod "N/A")
AORTA = 36
OUT = "/scratch/rnga/vvpshenov/DCE_NIK/results/xcat_physical_nomotion_nik_vs_grasp"

def _embed(a, N):                                    # center-embed the 152-dim into an N=220 grid (col-centered)
    a = a.T if a.shape[0] != N else a                # -> (N, 152)
    out = np.zeros((N, a.shape[0]) if False else (N, N), a.dtype)
    s = (N - a.shape[1]) // 2; out[:, s:s + a.shape[1]] = a
    return out

def load_slice(zi):
    """z-fft the aligned stack-of-stars to image-slice zi. returns per-slice 2D radial + geometry."""
    f = h5py.File(SIM, "r"); r = f["results"]; ks = r["kspace"]
    dce = np.array(ks["DCE"]["real"]) + 1j * np.array(ks["DCE"]["imag"])   # (F,C,112,RO)
    traj = np.array(ks["trajDCE"]); tim = np.array(r["images"]["Recon"]["timing"]).ravel()
    coil = np.array(ks["coilMaps"])                                        # (C,16,RO,RO) real
    F, C, S, RO = dce.shape
    npart = int(np.unique(np.round(traj[0, 2])).size); nang = S // npart    # 16, 7
    kd = np.zeros((C, F, nang, RO), np.complex64); kx = np.zeros((F, nang, RO), np.float32); ky = np.zeros((F, nang, RO), np.float32)
    for fr in range(F):
        blk = dce[fr].reshape(C, nang, npart, RO)
        prt = traj[fr, 2].reshape(nang, npart, RO)[:, :, 0].astype(int)
        tx = traj[fr, 0].reshape(nang, npart, RO); ty = traj[fr, 1].reshape(nang, npart, RO)
        for a in range(nang):
            srt = np.argsort(prt[a])
            zf = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(blk[:, a, srt, :], axes=1), axis=1), axes=1)
            kd[:, fr, a, :] = zf[:, zi, :]; kx[fr, a] = tx[a, srt][0]; ky[fr, a] = ty[a, srt][0]
    b1 = np.transpose(coil[:, zi], (1, 2, 0)).astype(np.complex64)         # (RO,RO,C)
    lab = _embed(np.array(r["labelGT"])[zi].astype(int), RO)               # (RO,RO)
    ref = _embed(np.abs(np.array(r["images"]["Recon"]["img"])[:, zi]).astype(np.float32).mean(0), RO)  # temporal-mean ref (sanity)
    return dict(kdata=kd, kx=kx, ky=ky, times=tim.astype(np.float32), b1=b1, labels=lab, RO=RO, nang=nang, npart=npart, ref=ref)

def truth_at(zi, t_query):
    """physical SPGR truth (magnitude), embedded [RO,RO,len(t_query)], sampled at query times (s)."""
    f = h5py.File(SIM, "r"); r = f["results"]
    gt = np.abs(np.array(r["images"]["GroundTruth"]["img"])[:, zi]).astype(np.float32)  # (901,RO,152)
    tg = np.array(r["images"]["GroundTruth"]["timing"]).ravel()
    RO = gt.shape[1]
    out = np.zeros((RO, RO, len(t_query)), np.float32)
    gtE = np.stack([_embed(gt[i], RO) for i in range(gt.shape[0])], 0)     # (901,RO,RO)
    for j, t in enumerate(t_query):
        i = np.searchsorted(tg, t); i0 = max(0, i - 1); i1 = min(len(tg) - 1, i)
        w = 0.0 if i1 == i0 else (t - tg[i0]) / (tg[i1] - tg[i0])
        out[:, :, j] = (1 - w) * gtE[i0] + w * gtE[i1]
    return out

CORTEX, MEDULLA = 13, 37   # XCAT renal cortex (label 13) + medulla (label 37, adjacent, strong delayed enhancement)
def rois(zi, labels=None):
    """aorta / cortex / medulla / organ / body masks in the recon grid (RO,RO)."""
    if labels is None: labels = load_slice(zi)["labels"]
    body = labels > 0; aorta = labels == AORTA
    return dict(aorta=aorta, cortex=(labels == CORTEX), medulla=(labels == MEDULLA),
                organ=(body & ~aorta), body=body, cortex_lab=CORTEX, medulla_lab=MEDULLA)
