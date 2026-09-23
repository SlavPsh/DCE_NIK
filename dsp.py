"""dataset paths for the in vivo pipeline. DCE_DS selects the scan: p3 (default, meas_p3_dce, every existing path unchanged) or p8 (meas_p8_dce,
same protocol: 1708 views, TA 375 s, nx 384 / bas 192, so spoke masks and TA carry over; only the data-derived artefacts are namespaced).
usage: DCE_DS=p8 python <script>; in code: import dsp; dsp.REF, dsp.STEP2(Z), ..."""
import os
DS = os.environ.get("DCE_DS", "p3"); SFX = "" if DS == "p3" else f"_{DS}"
D = "/net/beegfs/users/P101440/DCE_NIK"; PRO = "/net/beegfs/users/P101440/grasp_pro_py"; V2 = "/net/beegfs/users/P101440/grasp_v2"
RAW = {"p3": "/net/beegfs/users/P101440/dce_data/orig/meas_p3_dce.dat", "p8": "/net/beegfs/users/P101440/dce_data/orig/meas_p8_dce.dat"}[DS]
REF = f"{PRO}/results_ref{SFX}"                                              # precompute_ref output: shared.npz, slice_XX.npz (b1, cs_img, kdata_radial)
GV = f"{V2}/results_grasp_v2{SFX}"                                           # grasp (v2) recons gv2_sliceXX_<tag>.npy
GP = f"{PRO}/results_spoke_cs{SFX}"                                          # grasp-pro k80 recons cs_sliceXX_f80match.npy
TA = 375.0
def NUF(Z): return f"{D}/results_nufft{SFX}_slice{Z}"                        # build_rulers: nufft_all / nufft_pre / meta.json
def STEP2(Z): return f"{D}/step2{SFX}_slice{Z}.npz"                           # model-free 31-spoke series
def AIF(Z): return f"{D}/aif{SFX}_slice{Z}.npz"
def ROIS(Z): return f"{D}/results/realdata_nik_vs_cs_figures/rois_proposed{SFX}_sl{Z}.npz"
def SUPPORT(Z, d=6): return f"{D}/spoke_masks/support{SFX}_sl{Z}" + ("" if (d == 4 and DS == "p3") else f"_d{d}") + ".npy"
def BASIS(Z, r=None, rms1=True): return f"{D}/results/tofts_vs_patlak/basis{SFX}_sl{Z}" + (f"_r{r}" if r else "") + ("_rms1" if rms1 else "") + ".npz"
def GV_K80(Z): return f"{GV}/gv2_slice{Z}_n12_k80.npy"
def GP_K80(Z): return f"{GP}/cs_slice{Z}_f80match.npy"
FIGD = f"{D}/results/realdata_nik_vs_cs_figures"; RES = f"{D}/results/tofts_vs_patlak"
