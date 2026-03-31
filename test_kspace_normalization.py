"""
test_kspace_normalization.py — Acceptance tests for k-space normalization pipeline.

Run: python test_kspace_normalization.py
"""

import torch
import numpy as np

from kspace_normalization import (
    compute_radius,
    compute_dcf_radial,
    estimate_radial_envelope,
    compute_global_scale,
    KSpaceNormalizer,
    prepare_training_targets,
    prepare_validation_targets,
)
from losses import weighted_complex_mse, shell_balanced_complex_mse


def _make_synthetic_radial(n_spokes=64, n_ro=128, alpha=3.0, seed=42):
    """Create synthetic radial k-space with exponential decay envelope."""
    rng = torch.Generator().manual_seed(seed)
    angles = torch.linspace(0, torch.pi, n_spokes + 1)[:-1]
    s = torch.linspace(-0.5, 0.5, n_ro)

    kx_list, ky_list, y_list = [], [], []
    for theta in angles:
        kx = s * torch.cos(theta)
        ky = s * torch.sin(theta)
        r = torch.sqrt(kx ** 2 + ky ** 2)
        # Exponential decay * oscillatory structure
        envelope = torch.exp(-alpha * r)
        phi = torch.randn(n_ro, generator=rng) * 0.5 + 1.0  # ~unit scale
        y = envelope * phi * torch.exp(1j * torch.randn(n_ro, generator=rng))
        kx_list.append(kx)
        ky_list.append(ky)
        y_list.append(y)

    kcoords = torch.stack([torch.cat(kx_list), torch.cat(ky_list)], dim=1)
    y = torch.cat(y_list)
    return kcoords, y


def _make_synthetic_cartesian(nx=64, ny=64, alpha=3.0, seed=123):
    """Create synthetic Cartesian k-space."""
    kx_lin = torch.linspace(-0.5, 0.5, nx)
    ky_lin = torch.linspace(-0.5, 0.5, ny)
    KY, KX = torch.meshgrid(ky_lin, kx_lin, indexing="ij")
    kcoords = torch.stack([KX.reshape(-1), KY.reshape(-1)], dim=1)
    r = compute_radius(kcoords)
    rng = torch.Generator().manual_seed(seed)
    envelope = torch.exp(-alpha * r)
    phi = torch.randn(kcoords.shape[0], generator=rng) * 0.5 + 1.0
    y = envelope * phi * torch.exp(1j * torch.randn(kcoords.shape[0], generator=rng))
    return kcoords, y


def test_roundtrip_normalization():
    """Test 1: normalize then denormalize recovers original."""
    kcoords, y = _make_synthetic_radial()
    norm = KSpaceNormalizer()
    norm.fit(kcoords, y)

    y_tilde = norm.normalize(kcoords, y)
    y_rec = norm.denormalize(kcoords, y_tilde)

    rel_err = (y_rec - y).abs().max() / (y.abs().max() + 1e-12)
    print(f"Test 1 (round-trip): max relative error = {rel_err:.2e}", end="")
    assert rel_err < 1e-5, f"Round-trip error too large: {rel_err}"
    print(" ✓")


def test_roundtrip_real_format():
    """Test 1b: round-trip with (N, 2) real format."""
    kcoords, y_c = _make_synthetic_radial()
    y_real = torch.stack([y_c.real, y_c.imag], dim=-1)

    norm = KSpaceNormalizer()
    norm.fit(kcoords, y_real)

    y_tilde = norm.normalize(kcoords, y_real)
    assert y_tilde.shape == y_real.shape, "Output should be (N, 2) real"

    y_rec = norm.denormalize(kcoords, y_tilde)
    rel_err = (y_rec - y_real).abs().max() / (y_real.abs().max() + 1e-12)
    print(f"Test 1b (round-trip real): max relative error = {rel_err:.2e}", end="")
    assert rel_err < 1e-5
    print(" ✓")


def test_cartesian_same_normalizer():
    """Test 2: Cartesian uses the same normalizer fitted on radial."""
    kcoords_rad, y_rad = _make_synthetic_radial()
    kcoords_cart, y_cart = _make_synthetic_cartesian()

    norm = KSpaceNormalizer()
    norm.fit(kcoords_rad, y_rad)
    scale_before = norm.global_scale

    # Normalize Cartesian — should NOT refit
    y_cart_norm = norm.normalize(kcoords_cart, y_cart)
    assert norm.global_scale == scale_before, "Scale changed during Cartesian normalization!"
    print(f"Test 2 (same normalizer): scale unchanged ({scale_before:.4f}) ✓")


def test_dcf_properties():
    """Test 3: DCF weights are sensible."""
    kcoords, _ = _make_synthetic_radial()
    dcf = compute_dcf_radial(kcoords)

    r = compute_radius(kcoords)
    center_mask = r < 0.05
    outer_mask = r > 0.3

    center_mean = dcf[center_mask].mean().item()
    outer_mean = dcf[outer_mask].mean().item()
    dcf_mean = dcf.mean().item()

    print(f"Test 3 (DCF): center_w={center_mean:.3f}, outer_w={outer_mean:.3f}, "
          f"mean={dcf_mean:.4f}", end="")
    assert center_mean < outer_mean, "Center weights should be < outer weights"
    assert abs(dcf_mean - 1.0) < 0.01, f"DCF mean should be ~1, got {dcf_mean}"
    print(" ✓")


def test_envelope_flattens():
    """Test 4: Envelope normalization flattens radial trend."""
    kcoords, y = _make_synthetic_radial(alpha=5.0)
    r = compute_radius(kcoords)

    # Before: shell RMS should vary a lot (exponential decay)
    envelope = estimate_radial_envelope(kcoords, y, n_bins=32)
    a_r = envelope.evaluate(r)
    y_normed = y / (a_r + 1e-8)

    # Check shell RMS before and after
    n_shells = 8
    r_max = r.max().item()
    rms_before, rms_after = [], []
    for i in range(n_shells):
        lo = i * r_max / n_shells
        hi = (i + 1) * r_max / n_shells
        mask = (r >= lo) & (r < hi)
        if mask.sum() > 0:
            rms_before.append(y[mask].abs().pow(2).mean().sqrt().item())
            rms_after.append(y_normed[mask].abs().pow(2).mean().sqrt().item())

    cv_before = np.std(rms_before) / (np.mean(rms_before) + 1e-12)
    cv_after = np.std(rms_after) / (np.mean(rms_after) + 1e-12)
    print(f"Test 4 (envelope): CV before={cv_before:.3f}, after={cv_after:.3f}", end="")
    assert cv_after < cv_before, "Envelope should flatten radial trend"
    print(" ✓")


def test_weighted_loss_manual():
    """Test 5: Weighted loss matches manual computation."""
    pred = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j, 0.5 + 0.5j])
    target = torch.tensor([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j])
    weights = torch.tensor([1.0, 2.0, 3.0])
    power = 1.0

    err_sq = (pred - target).abs() ** 2  # [1.0, 1.0, 0.5]
    w_eff = weights ** power  # [1, 2, 3]
    expected = (w_eff * err_sq).sum() / w_eff.sum()

    loss = weighted_complex_mse(pred, target, weights=weights, power=power)
    diff = abs(loss.item() - expected.item())
    print(f"Test 5 (manual loss): expected={expected.item():.6f}, got={loss.item():.6f}, "
          f"diff={diff:.2e}", end="")
    assert diff < 1e-6
    print(" ✓")


def test_shell_balanced_loss():
    """Test 5b: Shell-balanced loss runs without error."""
    kcoords, y = _make_synthetic_radial()
    r = compute_radius(kcoords)
    pred = y + 0.1 * torch.randn_like(y)
    loss = shell_balanced_complex_mse(pred, y, r, n_bins=16)
    print(f"Test 5b (shell loss): loss={loss.item():.6f} ✓")


def test_prepare_api():
    """Test 6: prepare_training_targets and prepare_validation_targets work."""
    kcoords_rad, y_rad = _make_synthetic_radial()
    kcoords_cart, y_cart = _make_synthetic_cartesian()

    train_out = prepare_training_targets(kcoords_rad, y_rad)
    val_out = prepare_validation_targets(kcoords_cart, y_cart, train_out["normalizer"])

    assert "normalizer" in train_out
    assert "dcf" in train_out
    assert "y_radial_norm" in train_out
    assert "y_cart_norm" in val_out
    print(f"Test 6 (prepare API): train_norm shape={train_out['y_radial_norm'].shape}, "
          f"val_norm shape={val_out['y_cart_norm'].shape} ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("K-space normalization acceptance tests")
    print("=" * 60)
    test_roundtrip_normalization()
    test_roundtrip_real_format()
    test_cartesian_same_normalizer()
    test_dcf_properties()
    test_envelope_flattens()
    test_weighted_loss_manual()
    test_shell_balanced_loss()
    test_prepare_api()
    print("=" * 60)
    print("All tests passed!")
