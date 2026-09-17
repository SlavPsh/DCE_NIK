"""body support mask on the full nx grid for the k-space support prior (train_grasp_nik --support-mask): the approved body mask of the slice
(consolidated.slice_ctx BODY, bas x bas, recon_nik_cart orientation) dilated by a safety margin and zero-padded to nx x nx at the centre.
usage: python support_mask.py --slices 21,18,19 --dilate 4  -> spoke_masks/support_sl<Z>.npy (bool [nx,nx])"""
import sys, argparse, numpy as np, scipy.ndimage as ndi
D = "/net/beegfs/users/P101440/DCE_NIK"; sys.path.insert(0, D)
import consolidated as C
ap = argparse.ArgumentParser(); ap.add_argument("--slices", default="21,18,19"); ap.add_argument("--dilate", type=int, default=4); a = ap.parse_args()
sh = np.load("/net/beegfs/users/P101440/grasp_pro_py/results_ref/shared.npz"); nx = int(sh["nx"]); bas = int(sh["bas"]); s = (nx - bas) // 2
for Z in [int(z) for z in a.slices.split(",")]:
    body = ndi.binary_dilation(C.slice_ctx(Z)["BODY"], iterations=a.dilate); full = np.zeros((nx, nx), bool); full[s:s + bas, s:s + bas] = body
    np.save(f"{D}/spoke_masks/support_sl{Z}.npy", full); print(f"slice {Z}: body {int(body.sum())} px of {bas}x{bas}, support {int(full.sum())} px of {nx}x{nx} ({100 * full.mean():.1f}%)")
