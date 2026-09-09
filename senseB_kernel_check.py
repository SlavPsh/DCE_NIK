"""SENSE-B feasibility: compactness of the coil kernel B_c = FT(b1_c).
per-coil prediction = (X * B_c)(k), so every kernel tap costs one extra network evaluation."""
import warnings; warnings.filterwarnings("ignore")
import sys, numpy as np
sys.path.insert(0, "/net/beegfs/users/P101440/grasp_pro_py")
from fftc import fft2c_mri
REF = "/net/beegfs/users/P101440/grasp_pro_py/results_ref"
b1 = np.asarray(np.load(f"{REF}/slice_21.npz")["b1"]); b1 = b1 / np.abs(b1).max()
K = fft2c_mri(b1)                       # [x,y,C] in, FFT over the two SPATIAL axes
E = (np.abs(K) ** 2).sum(-1); tot = E.sum(); c0 = E.shape[0] // 2
print(f"{'r':>3} {'taps':>6} {'energy %':>9}")
for r in (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48):
    print(f"{r:>3} {(2*r+1)**2:>6} {100*E[c0-r:c0+r+1, c0-r:c0+r+1].sum()/tot:>8.2f}%")
