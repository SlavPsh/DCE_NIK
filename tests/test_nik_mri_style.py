#!/usr/bin/env python3
"""nikmri style tests"""

import torch

from nik_loss import HDRLossFF
from nik_model import NIK_MRI_SIREN_REIM
from nik_mri_style import (
    DynamicSliceDataset,
    build_nik_mri_dce_dataset,
    normalized_time_coords,
    predict_cartesian_kspace,
)


def test_model_forward():
    model = NIK_MRI_SIREN_REIM(
        coord_dim=4,
        feature_dim=32,
        num_layers=3,
        out_dim=1,
        omega_0=30.0,
        ff_seed=0,
    )
    coords = torch.randn(64, 4)
    out = model(coords)
    assert torch.is_complex(out)
    assert out.shape == (64, 1)
    print("model forward ok")


def test_hdr_loss():
    loss_fn = HDRLossFF(sigma=1.0, eps=1e-2, factor=0.1)
    coords = torch.randn(128, 4)
    pred = torch.randn(128, 1, dtype=torch.complex64)
    target = torch.randn(128, 1, dtype=torch.complex64)
    loss, reg = loss_fn(pred, target, coords)
    assert loss.ndim == 0
    assert reg.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(reg)
    print("hdr loss ok")


def test_dynamic_dataset_builder():
    T, S_per_slice, C, Z, RO = 3, 4, 2, 2, 5
    S = S_per_slice * Z

    k_img_space = torch.randn(T, S_per_slice, C, Z, RO, dtype=torch.complex64)
    traj_t = torch.randn(T, S, 3, RO, dtype=torch.float32)
    scales = (
        torch.tensor(1.0),
        torch.tensor(1.0),
        torch.tensor(1.0),
    )

    coords, targets, meta = build_nik_mri_dce_dataset(
        k_img_space,
        traj_t,
        scales,
        z_slice_idx=1,
        n_slices=Z,
        target_device="cpu",
    )
    expected = T * C * S_per_slice * RO

    assert coords.shape == (expected, 4)
    assert targets.shape == (expected, 1)
    assert torch.allclose(torch.abs(targets).amax(), torch.tensor(1.0), atol=1e-6)
    assert meta["n_total"] == expected

    ds = DynamicSliceDataset(coords, targets)
    sample = ds[0]
    assert sample["coords"].shape == (4,)
    assert sample["targets"].shape == (1,)
    print("dataset builder ok")


def test_cartesian_prediction():
    model = NIK_MRI_SIREN_REIM(
        coord_dim=4,
        feature_dim=16,
        num_layers=2,
        out_dim=1,
        ff_seed=0,
    )
    kpred = predict_cartesian_kspace(
        model,
        nt=2,
        nc=3,
        nx=8,
        ny=6,
        device="cpu",
        chunk_size=16,
        time_coords=normalized_time_coords(2),
        coil_coords=torch.linspace(-1.0, 1.0, 3),
    )
    assert kpred.shape == (2, 3, 8, 6)
    assert torch.is_complex(kpred)
    print("cartesian prediction ok")


if __name__ == "__main__":
    test_model_forward()
    test_hdr_loss()
    test_dynamic_dataset_builder()
    test_cartesian_prediction()
    print("all nik_mri_style tests passed")
