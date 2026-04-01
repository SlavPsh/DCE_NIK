import torch

from nik_recon import make_cartesian_eval_dataset


def test_make_cartesian_eval_dataset_uses_centered_fft_bins():
    k_cart_z = torch.zeros((1, 1, 1, 4, 6), dtype=torch.complex64)

    x_cart, y_cart, meta = make_cartesian_eval_dataset(
        k_cart_z,
        t_fixed=0,
        coil_fixed=0,
        z_slice_idx=0,
        scales_radial=(1.0, 1.0, 1.0),
        y_scale=torch.tensor(1.0),
        compute_device="cpu",
    )

    coords = x_cart.reshape(meta["nky"], meta["nkx"], 2)

    expected_kx = torch.tensor([-0.5, -1.0 / 3.0, -1.0 / 6.0, 0.0, 1.0 / 6.0, 1.0 / 3.0])
    expected_ky = torch.tensor([-0.5, -0.25, 0.0, 0.25])

    assert torch.allclose(coords[0, :, 0], expected_kx, atol=1e-6)
    assert torch.allclose(coords[:, 0, 1], expected_ky, atol=1e-6)
    assert torch.allclose(coords[meta["nky"] // 2, meta["nkx"] // 2], torch.zeros(2), atol=1e-6)
    assert y_cart.shape == (meta["nky"] * meta["nkx"], 2)
