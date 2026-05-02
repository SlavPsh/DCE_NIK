#!/usr/bin/env python3
"""perceptual metrics validation"""
import numpy as np
import sys


def shepp_logan_phantom(size=128):
    """shepp logan phantom"""
    img = np.zeros((size, size), dtype=np.float64)
    y, x = np.mgrid[-1:1:size*1j, -1:1:size*1j]

    # outer ellipse
    mask = (x / 0.69)**2 + (y / 0.92)**2 <= 1
    img[mask] = 1.0

    # inner dark ellipse
    mask = (x / 0.6624)**2 + (y / 0.874)**2 <= 1
    img[mask] = 0.2

    # bright ellipses
    mask = ((x - 0.22) / 0.11)**2 + (y / 0.31)**2 <= 1
    img[mask] = 0.8
    mask = ((x + 0.22) / 0.16)**2 + (y / 0.41)**2 <= 1
    img[mask] = 0.6

    return img


def main():
    from nik_metrics import (
        compute_image_metrics,
        compute_perceptual_metrics,
        format_metrics_table,
        compare_reconstructions,
    )

    print("Generating test images...")
    img_ref = shepp_logan_phantom(128)

    # blurred
    from scipy.ndimage import gaussian_filter
    img_blurred = gaussian_filter(img_ref, sigma=2.0)

    # noisy
    rng = np.random.default_rng(42)
    img_noisy = img_ref + rng.normal(0, 0.05, img_ref.shape)
    img_noisy = np.clip(img_noisy, 0, None)

    # identical
    img_identical = img_ref.copy()

    print("\n=== Basic metrics (blurred vs ref) ===")
    basic = compute_image_metrics(img_blurred, img_ref)
    print(format_metrics_table(basic))

    print("\n=== Perceptual metrics (blurred vs ref) ===")
    perc = compute_perceptual_metrics(img_blurred, img_ref)
    print(format_metrics_table(perc))

    print("\n=== Perceptual metrics (noisy vs ref) ===")
    perc_noisy = compute_perceptual_metrics(img_noisy, img_ref)
    print(format_metrics_table(perc_noisy))

    print("\n=== Perceptual metrics (identical vs ref) ===")
    perc_id = compute_perceptual_metrics(img_identical, img_ref)
    print(format_metrics_table(perc_id))

    # comparison table
    print("\n=== Comparison table ===")
    all_metrics = [
        {**compute_image_metrics(img_identical, img_ref),
         **compute_perceptual_metrics(img_identical, img_ref)},
        {**compute_image_metrics(img_blurred, img_ref),
         **compute_perceptual_metrics(img_blurred, img_ref)},
        {**compute_image_metrics(img_noisy, img_ref),
         **compute_perceptual_metrics(img_noisy, img_ref)},
    ]
    labels = ["Identical", "Blurred", "Noisy"]
    print(compare_reconstructions(all_metrics, labels))

    # sanity checks
    print("\n=== Sanity checks ===")
    ok = True

    # identical scores
    if perc_id["PSNR"] < 50:
        print(f"FAIL: identical PSNR={perc_id['PSNR']:.1f}, expected >50")
        ok = False
    if perc_id["DISTS"] > 0.01:
        print(f"FAIL: identical DISTS={perc_id['DISTS']:.4f}, expected ~0")
        ok = False

    # blurred worse
    if perc["DISTS"] <= perc_id["DISTS"]:
        print(f"FAIL: blurred DISTS should be > identical DISTS")
        ok = False

    # positive distance
    for name in ["DISTS"]:
        for label, m in zip(["blurred", "noisy"], [perc, perc_noisy]):
            if m[name] < 0:
                print(f"FAIL: {label} {name}={m[name]:.4f}, expected >= 0")
                ok = False

    if ok:
        print("All sanity checks passed.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
