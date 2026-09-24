"""dataset paths for the in vivo pipeline. DCE_DS selects the scan: p3 (default, meas_p3_dce, every existing path unchanged), p8 (meas_p8_dce,
truncated copy, not used), p14 (meas_topqmri_p14: base 256, TR 5.95, flip 15, 2172 views, TA 390 s, 36 partitions). geometry (TA, nx, bas,
ntviews) is read from the dataset's shared.npz when it exists; the spoke masks (keep v%10<8 / val 8 / test 9) are per dataset.
usage: DCE_DS=p14 python <script>; in code: import dsp; dsp.REF, dsp.STEP2(Z), dsp.TA, dsp.KEEP, ..."""
import os, numpy as np
DS = os.environ.get("DCE_DS", "p3"); SFX = "" if DS == "p3" else f"_{DS}"
D = "/net/beegfs/users/P101440/DCE_NIK"; PRO = "/net/beegfs/users/P101440/grasp_pro_py"; V2 = "/net/beegfs/users/P101440/grasp_v2"
RAW = {"p3": "/net/beegfs/users/P101440/dce_data/orig/meas_p3_dce.dat", "p8": "/net/beegfs/users/P101440/dce_data/orig/meas_p8_dce.dat", "p14": "/net/beegfs/users/P101440/dce_data/orig/meas_topqmri_p14.dat"}[DS]
REF = f"{PRO}/results_ref{SFX}"                                              # precompute_ref output: shared.npz, slice_XX.npz (b1, cs_img, kdata_radial)
GV = f"{V2}/results_grasp_v2{SFX}"                                           # grasp (v2) recons gv2_sliceXX_<tag>.npy
GP = f"{PRO}/results_spoke_cs{SFX}"                                          # grasp-pro k80 recons cs_sliceXX_f80match.npy
def _geom():
    p = f"{REF}/shared.npz"
    if os.path.exists(p):
        sh = np.load(p); return dict(TA=float(sh["TA"]), nx=int(sh["nx"]), bas=int(sh["bas"]), ntviews=int(sh["ntviews"]))
    return dict(TA=375.0, nx=384, bas=192, ntviews=1708)
GEOM = _geom(); TA = GEOM["TA"]; NX = GEOM["nx"]; BAS = GEOM["bas"]; NTV = GEOM["ntviews"]
# spoke masks: p3 keeps its historical names (1708 views); other datasets get keep_k80_<ds>.npy etc. built by spoke_masks_build.py from ntviews
KEEP = f"{D}/spoke_masks/keep_f80match.npy" if DS == "p3" else f"{D}/spoke_masks/keep_k80{SFX}.npy"
VAL = f"{D}/spoke_masks/val_k80_m8.npy" if DS == "p3" else f"{D}/spoke_masks/val_k80_m8{SFX}.npy"
TEST = f"{D}/spoke_masks/test_k80_m9.npy" if DS == "p3" else f"{D}/spoke_masks/test_k80_m9{SFX}.npy"
def NUF(Z): return f"{D}/results_nufft{SFX}_slice{Z}"                        # build_rulers: nufft_all / nufft_pre / meta.json
def STEP2(Z): return f"{D}/step2{SFX}_slice{Z}.npz"                           # model-free 31-spoke series
def AIF(Z): return f"{D}/aif{SFX}_slice{Z}.npz"
def ROIS(Z): return f"{D}/results/realdata_nik_vs_cs_figures/rois_proposed{SFX}_sl{Z}.npz"
def SUPPORT(Z, d=6): return f"{D}/spoke_masks/support{SFX}_sl{Z}" + ("" if (d == 4 and DS == "p3") else f"_d{d}") + ".npy"
def BASIS(Z, r=None, rms1=True): return f"{D}/results/tofts_vs_patlak/basis{SFX}_sl{Z}" + (f"_r{r}" if r else "") + ("_rms1" if rms1 else "") + ".npz"
def GV_K80(Z): return f"{GV}/gv2_slice{Z}_n12_k80.npy"
def GP_K80(Z): return f"{GP}/cs_slice{Z}_f80match.npy"
FIGD = f"{D}/results/realdata_nik_vs_cs_figures"; RES = f"{D}/results/tofts_vs_patlak"
# tissue rois per dataset: p3 = kidney (cortex / medulla, approved 2026-09-15); p14 = liver / spleen (user choice 2026-09-24). the static roi used for
# temporal-noise readouts is the posterior 'liver' mask on p3 (historical name) and 'static' elsewhere.
T1, T2 = ("cortex", "medulla") if DS == "p3" else ("liver", "spleen")
ROI_NAMES = ("aorta", T1, T2); STATIC = "liver" if DS == "p3" else "static"
