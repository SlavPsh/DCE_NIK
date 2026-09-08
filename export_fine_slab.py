"""fine-time (1.1s, 342-frame) CS NIfTI slab for the slices with k-space (18-21). the native
cs_img IS the CS subspace render at 1.1s (3x finer than the 3s volume) -> resolves the
respiratory wobble the 3s binning averaged out. axial-over-time is the useful view (slab is
only 4 slices thick, so coronal/sagittal are thin). out: nifti_export/cs_dynamic_4d_fine_slab.nii.gz + labels."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, nibabel as nib, sys
sys.path.insert(0, "/scratch/rnga/vvpshenov/grasp_pro_py"); sys.path.insert(0, ".")
import consolidated as C
REF = "/scratch/rnga/vvpshenov/grasp_pro_py/results_ref"; OUT = "/scratch/rnga/vvpshenov/DCE_NIK/nifti_export"
SLAB = [18, 19, 20, 21]; ROI_SL = [18, 19, 21]; LAB = {"aorta": 1, "cortex": 2, "medulla": 3, "liver": 4}
VOX = (3.0, 3.0, 4.0)
cubes = [np.abs(np.load(f"{REF}/slice_{z:02d}.npz")["cs_img"]).astype(np.float32) for z in SLAB]  # each [192,192,342]
vol = np.stack(cubes, axis=2)                                                     # [192,192,Zslab,342]
X, Y, Zs, T = vol.shape
lab = np.zeros((X, Y, Zs), np.int16)
for zi, z in enumerate(SLAB):
    if z in ROI_SL:
        rois = C.slice_ctx(z)["rois"]
        for nm, v in LAB.items():
            m = rois.get(nm)
            if m is not None and m.sum() > 0: lab[:, :, zi][m] = v
aff = np.diag([VOX[0], VOX[1], VOX[2], 1.0])
img = nib.Nifti1Image(vol, aff); img.header["pixdim"][4] = 375.0 / (T - 1)         # ~1.1s TR (frame dt)
nib.save(img, f"{OUT}/cs_dynamic_4d_fine_slab.nii.gz")
nib.save(nib.Nifti1Image(lab, aff), f"{OUT}/cs_roi_labels_fine_slab.nii.gz")
print(f"wrote cs_dynamic_4d_fine_slab {vol.shape} (Z={SLAB}, dt={375.0/(T-1):.2f}s) + labels -> {OUT}")
print(f"slab z-index -> slice: " + ", ".join(f"{i}->sl{z}" for i, z in enumerate(SLAB)))
