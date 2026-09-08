"""k-space-native SPIRiT prior. calibrate G from the dense radial centre, penalize
||(G - I) x||^2 on a cartesian patch during training. no NUFFT, no render, no image domain.

G is a coil-consistency operator: each cartesian k-point is predicted from its
kxk multi-coil neighborhood. (G-I)x = 0 for coil-correlation-consistent k-space, so the
penalty suppresses aliasing that violates coil correlations. it is LINEAR, it does NOT do
edge-preserving sparsity like image-domain TV."""
import numpy as np
import torch
import torch.nn.functional as F


def calibrate_spirit(acs, ksize=5, lam=1e-2):
    """acs [C, N, N] complex cartesian calibration block -> G [C, C, ksize, ksize] complex.
    G[co] maps the kxk neighborhood of all coils to coil co's center; center-of-self excluded."""
    C, N, N2 = acs.shape
    k = ksize
    kc = k // 2
    # sliding kxk patches over the ACS, all coils
    patches = []
    for i in range(N - k + 1):
        for j in range(N2 - k + 1):
            patches.append(acs[:, i:i + k, j:j + k].reshape(-1))     # [C*k*k]
    A = np.stack(patches, 0)                                          # [P, C*k*k]
    center = (np.arange(C) * k * k) + (kc * k + kc)                   # self-center index per coil
    G = np.zeros((C, C, k, k), np.complex128)
    AhA = A.conj().T @ A
    reg = lam * np.trace(AhA).real / AhA.shape[0] * np.eye(AhA.shape[0])
    for co in range(C):
        src = A.copy(); src[:, center[co]] = 0                        # exclude the target point itself
        tgt = A[:, center[co]]                                        # coil co center
        AhA_c = src.conj().T @ src + reg
        g = np.linalg.solve(AhA_c, src.conj().T @ tgt)               # [C*k*k]
        G[co] = g.reshape(C, k, k)
    return G.astype(np.complex64)


def spirit_penalty(pred_patch, G_re, G_im):
    """pred_patch [C, 2, P, P] (re/im), G_* [C, C, k, k]. returns mean |(G-I)x|^2.
    conv2d with the calibrated kernel, complex, then residual against the input."""
    C = pred_patch.shape[0]
    xr = pred_patch[:, 0][None]                                       # [1, C, P, P]
    xi = pred_patch[:, 1][None]
    pad = G_re.shape[-1] // 2
    # complex conv: (Gr+iGi)(xr+ixi)
    yr = F.conv2d(xr, G_re, padding=pad) - F.conv2d(xi, G_im, padding=pad)
    yi = F.conv2d(xr, G_im, padding=pad) + F.conv2d(xi, G_re, padding=pad)
    rr = yr[0] - pred_patch[:, 0]                                     # (G - I) x, real
    ri = yi[0] - pred_patch[:, 1]
    return (rr ** 2 + ri ** 2).mean()
