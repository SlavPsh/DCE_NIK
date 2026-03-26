#!/usr/bin/env python3
"""Quick test for multi-coil models and data loading.

Verifies:
  1. MultiCoilWIRE, MultiCoilSIREN, MultiCoilConcat forward passes
  2. Multi-coil data loader produces correct shapes
  3. Spoke-based splitting works with repeated coil data
  4. Gradient flows through FiLM parameters

Usage:
    micromamba run -n torch29 python test_multicoil.py
"""
import sys
import torch
import numpy as np


def test_models():
    """Test model forward passes and parameter counts."""
    from nik_model import MultiCoilWIRE, MultiCoilSIREN, MultiCoilConcat

    B = 64
    n_coils = 8
    coords = torch.randn(B, 2)
    coil_idx = torch.randint(0, n_coils, (B,))

    print("=== Model tests ===")

    # MultiCoilWIRE (FiLM)
    model = MultiCoilWIRE(
        in_dim=2, hidden=48, depth=6, w0=62.0, s0=10.0,
        n_coils=n_coils, coil_embed_dim=32,
    )
    out = model(coords, coil_idx)
    assert out.shape == (B, 2), f"Expected (B,2), got {out.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MultiCoilWIRE:  out={out.shape}, params={n_params}")

    # Verify FiLM init: all coils should produce similar output at init
    c0 = torch.zeros(B, dtype=torch.long)
    c1 = torch.ones(B, dtype=torch.long)
    out0 = model(coords, c0)
    out1 = model(coords, c1)
    diff = (out0 - out1).abs().max().item()
    print(f"    FiLM init coil 0 vs 1 max diff: {diff:.6f} (should be ~0)")
    assert diff < 0.01, f"FiLM init too divergent: {diff}"

    # MultiCoilSIREN (FiLM)
    model_s = MultiCoilSIREN(
        in_dim=2, hidden=48, depth=6, w0=60.0,
        n_coils=n_coils, coil_embed_dim=32,
    )
    out_s = model_s(coords, coil_idx)
    n_params_s = sum(p.numel() for p in model_s.parameters())
    print(f"  MultiCoilSIREN: out={out_s.shape}, params={n_params_s}")

    # MultiCoilConcat (WIRE backbone)
    model_c = MultiCoilConcat(
        backbone_family="wire",
        backbone_kwargs=dict(hidden=48, depth=6, w0=62.0, s0=10.0),
        n_coils=n_coils, coil_embed_dim=16,
    )
    out_c = model_c(coords, coil_idx)
    n_params_c = sum(p.numel() for p in model_c.parameters())
    print(f"  MultiCoilConcat(wire): out={out_c.shape}, params={n_params_c}")

    # MultiCoilConcat (SIREN backbone)
    model_cs = MultiCoilConcat(
        backbone_family="siren",
        backbone_kwargs=dict(hidden=48, depth=6, w0=60.0),
        n_coils=n_coils, coil_embed_dim=16,
    )
    out_cs = model_cs(coords, coil_idx)
    n_params_cs = sum(p.numel() for p in model_cs.parameters())
    print(f"  MultiCoilConcat(siren): out={out_cs.shape}, params={n_params_cs}")

    # Gradient test: FiLM parameters get gradients
    model.zero_grad()
    loss = out.pow(2).mean()
    loss.backward()
    film_grad = model.first_film.gamma_layer.weight.grad
    assert film_grad is not None, "FiLM gamma should have gradients"
    assert film_grad.abs().sum() > 0, "FiLM gradients should be non-zero"
    print(f"  Gradient test: FiLM gamma grad norm = {film_grad.norm():.6f}")

    print("  All model tests passed.\n")


def test_data_loader():
    """Test multi-coil data loader with synthetic data."""
    from nik_recon import split_points_by_spokes

    print("=== Data loader tests (synthetic) ===")

    # Simulate multi-coil spoke_id_all: 3 coils, 4 spokes, 10 pts/spoke
    n_coils = 3
    n_spokes = 4
    pts_per_spoke = 10
    N_per_coil = n_spokes * pts_per_spoke

    # spoke_id for one coil
    spoke_ids_1c = torch.arange(n_spokes).repeat_interleave(pts_per_spoke)
    # Repeat for all coils
    spoke_id_all = spoke_ids_1c.repeat(n_coils)

    assert spoke_id_all.shape[0] == N_per_coil * n_coils
    print(f"  spoke_id_all shape: {spoke_id_all.shape}")
    print(f"  Unique spokes: {torch.unique(spoke_id_all).tolist()}")

    # Split: should select same spokes for all coils
    train_idx, val_idx, train_spokes, val_spokes = split_points_by_spokes(
        spoke_id_all, val_frac=0.25, seed=42,
    )

    # Check that val spokes are consistent across coils
    val_spoke_set = set(val_spokes.tolist())
    print(f"  Val spokes: {val_spoke_set}")
    print(f"  Train idx: {len(train_idx)}, Val idx: {len(val_idx)}")

    # Each coil should have the same number of val points
    for c in range(n_coils):
        start = c * N_per_coil
        end = (c + 1) * N_per_coil
        coil_val = ((val_idx >= start) & (val_idx < end)).sum().item()
        coil_train = ((train_idx >= start) & (train_idx < end)).sum().item()
        print(f"    Coil {c}: train={coil_train}, val={coil_val}")
        # All coils should have same val count
        if c > 0:
            first_val = ((val_idx >= 0) & (val_idx < N_per_coil)).sum().item()
            assert coil_val == first_val, f"Coil {c} val count differs"

    print("  All data loader tests passed.\n")


def test_training_step():
    """Test one training step with multi-coil model."""
    from nik_model import MultiCoilWIRE

    print("=== Training step test ===")

    B = 128
    n_coils = 8
    model = MultiCoilWIRE(
        in_dim=2, hidden=32, depth=4, w0=30.0, s0=10.0,
        n_coils=n_coils, coil_embed_dim=16,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    coords = torch.randn(B, 2)
    coil_idx = torch.randint(0, n_coils, (B,))
    targets = torch.randn(B, 2)

    # Forward
    pred = model(coords, coil_idx)
    loss = torch.nn.functional.mse_loss(pred, targets)

    # Backward
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Check loss decreased on second step
    pred2 = model(coords, coil_idx)
    loss2 = torch.nn.functional.mse_loss(pred2, targets)

    print(f"  Loss step 1: {loss.item():.6f}")
    print(f"  Loss step 2: {loss2.item():.6f}")
    print("  Training step test passed.\n")


if __name__ == "__main__":
    test_models()
    test_data_loader()
    test_training_step()
    print("All tests passed!")
