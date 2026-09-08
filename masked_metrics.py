"""Window metrics evaluated ONLY over a body mask, with no artificial mask edge.

Masking the *input* (img*mask) is wrong: it pastes a hard step at the boundary that is
identical in both images, and HaarPSI weights by edge strength -> large inflation.
Leaving the background in is also wrong: it scores agreement on streaks/air.
Correct: build the per-pixel similarity/weight maps, then average ONLY over the mask.

haarpsi_masked replicates piq.haarpsi exactly and restricts its two sums to the mask.
ssim_masked builds the standard SSIM map and averages it over the mask.
Both are validated against piq with an all-ones mask (see __main__)."""
import torch
import torch.nn.functional as F
from piq.functional import haar_filter, similarity_map


def _pool_mask(mask, subsample=True):
    if not subsample:
        return mask
    down = max(mask.shape[2] % 2, mask.shape[3] % 2)
    mask = F.pad(mask, pad=[0, down, 0, down])
    return F.avg_pool2d(mask, kernel_size=2, stride=2, padding=0)


def haarpsi_masked(x, y, mask=None, data_range=1., scales=3, subsample=True,
                   c=30.0, alpha=4.2):
    """piq.haarpsi with the weighted sums restricted to `mask` (soft, pooled to match)."""
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255
    if subsample:
        down = max(x.shape[2] % 2, x.shape[3] % 2)
        pad_to_use = [0, down, 0, down]
        x = F.avg_pool2d(F.pad(x, pad=pad_to_use), kernel_size=2, stride=2)
        y = F.avg_pool2d(F.pad(y, pad=pad_to_use), kernel_size=2, stride=2)

    cx, cy = [], []
    for scale in range(scales):
        ks = 2 ** (scale + 1)
        hf = haar_filter(ks, dtype=x.dtype, device=x.device)
        kernels = torch.stack([hf, hf.transpose(-1, -2)])
        pad_to_use = [ks // 2 - 1, ks // 2, ks // 2 - 1, ks // 2]
        cx.append(F.conv2d(F.pad(x[:, :1], pad=pad_to_use, mode='constant'), kernels))
        cy.append(F.conv2d(F.pad(y[:, :1], pad=pad_to_use, mode='constant'), kernels))
    cx, cy = torch.cat(cx, dim=1), torch.cat(cy, dim=1)

    weights = torch.max(torch.abs(cx[:, 4:]), torch.abs(cy[:, 4:]))
    sim = []
    for o in range(2):
        mx = torch.abs(cx[:, (o, o + 2)]); my = torch.abs(cy[:, (o, o + 2)])
        sim.append(similarity_map(mx, my, constant=c).sum(dim=1, keepdims=True) / 2)
    sim = torch.cat(sim, dim=1)

    if mask is None:
        m = torch.ones_like(weights[:, :1])
    else:
        m = _pool_mask(mask.to(x.dtype), subsample)
        m = m[..., :weights.shape[-2], :weights.shape[-1]]
    eps = torch.finfo(sim.dtype).eps
    num = ((sim * alpha).sigmoid() * weights * m).sum(dim=[1, 2, 3]) + eps
    den = (weights * m).sum(dim=[1, 2, 3]) + eps
    s = num / den
    return ((torch.log(s / (1 - s)) / alpha) ** 2)


def ssim_masked(x, y, mask=None, data_range=1., kernel_size=11, sigma=1.5):
    """standard SSIM map, averaged over `mask` only."""
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - (kernel_size - 1) / 2.
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :]).view(1, 1, kernel_size, kernel_size)
    pad = kernel_size // 2

    def flt(z):
        return F.conv2d(F.pad(z, [pad] * 4, mode='reflect'), k)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mx, my = flt(x), flt(y)
    mxx, myy, mxy = flt(x * x), flt(y * y), flt(x * y)
    vx, vy, vxy = mxx - mx * mx, myy - my * my, mxy - mx * my
    smap = ((2 * mx * my + C1) * (2 * vxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))
    if mask is None:
        return smap.mean(dim=[1, 2, 3])
    m = mask.to(x.dtype)
    return (smap * m).sum(dim=[1, 2, 3]) / m.sum(dim=[1, 2, 3]).clamp_min(1e-12)


if __name__ == "__main__":
    import numpy as np, piq
    torch.manual_seed(0)
    a = torch.rand(1, 1, 192, 192); b = (a + 0.15 * torch.rand_like(a)).clamp(0, 1)
    ones = torch.ones_like(a)
    print("validation (all-ones mask must reproduce piq):")
    print(f"  haarpsi  piq {float(piq.haarpsi(a, b, data_range=1.)):.6f}   "
          f"mine {float(haarpsi_masked(a, b, ones, data_range=1.)):.6f}")
    print(f"  ssim     piq {float(piq.ssim(a, b, data_range=1., downsample=False)):.6f}   "
          f"mine {float(ssim_masked(a, b, ones, data_range=1.)):.6f}")
