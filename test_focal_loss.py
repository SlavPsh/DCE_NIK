"""focal loss unit tests"""
import sys
import torch
import torch.nn.functional as F

from nik_focal_loss import composable_kspace_loss, _residual_magsq
from losses import weighted_complex_mse


def _make_inputs(n=4096, c=8, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    y_target = torch.randn(n, 2 * c, generator=g, device=device)
    y_pred   = y_target + 0.1 * torch.randn(n, 2 * c, generator=g, device=device)
    kx       = torch.randn(n, generator=g, device=device)
    ky       = torch.randn(n, generator=g, device=device)
    dcf      = torch.sqrt(kx ** 2 + ky ** 2)
    dcf      = dcf / dcf.mean()
    return y_pred, y_target, dcf


def test_gradient_isolation():
    """w_focal must have no gradient flowing into it"""
    y_pred, y_target, dcf = _make_inputs()
    y_pred = y_pred.requires_grad_(True)
    loss = composable_kspace_loss(
        y_pred, y_target, dcf=dcf, use_dcf=True, dcf_power=1.0,
        use_focal=True, focal_alpha=1.0,
    )
    loss.backward()
    assert y_pred.grad is not None and torch.isfinite(y_pred.grad).all(), \
        "expected finite grads on y_pred"
    print("PASS gradient_isolation: y_pred.grad finite, no NaN in backward")


def test_toggle_independence_off():
    """all toggles off reduces to plain MSE on real-interleaved layout"""
    y_pred, y_target, dcf = _make_inputs()
    loss = composable_kspace_loss(
        y_pred, y_target, dcf=dcf, use_dcf=False, use_focal=False,
    )
    expected = (y_pred - y_target).pow(2).sum(dim=-1).mean()
    assert torch.isclose(loss, expected, rtol=1e-5), \
        f"plain MSE mismatch: got {float(loss)} vs {float(expected)}"
    print(f"PASS toggle_independence_off: loss={float(loss):.6e}")


def test_envelope_only_equivalence():
    """matches existing weighted_complex_mse when focal=off, dcf=on, no envelope"""
    y_pred, y_target, dcf = _make_inputs()
    # existing pipeline call
    expected = weighted_complex_mse(y_pred, y_target, weights=dcf, power=1.0, normalize=True)
    actual = composable_kspace_loss(
        y_pred, y_target, dcf=dcf, use_dcf=True, dcf_power=1.0,
        use_focal=False,
    )
    rtol = 1e-5
    assert torch.isclose(actual, expected, rtol=rtol), \
        f"envelope-only equivalence broken: new {float(actual):.6e} vs old {float(expected):.6e}"
    print(f"PASS envelope_only_equivalence: actual={float(actual):.6e} expected={float(expected):.6e}")


def test_complex_gradient_flow():
    """small synthetic complex optimization converges"""
    n, c = 1024, 2
    torch.manual_seed(0)
    y_target_c = torch.randn(n, c, dtype=torch.complex64)
    target = torch.view_as_real(y_target_c).reshape(n, 2 * c)
    w = torch.nn.Parameter(torch.zeros(n, 2 * c))
    opt = torch.optim.Adam([w], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        loss = composable_kspace_loss(
            w, target, use_focal=True, focal_alpha=1.0,
            focal_warmup_progress=1.0,
        )
        loss.backward()
        opt.step()
    final = float(loss.item())
    assert final < 1e-4, f"complex_gradient_flow did not converge: final={final}"
    print(f"PASS complex_gradient_flow: final loss {final:.3e}")


def test_warmup_ramp():
    """warmup progress=0 -> uniform weights; progress=1 -> focal weights"""
    y_pred, y_target, dcf = _make_inputs()
    loss_warmup0 = composable_kspace_loss(
        y_pred, y_target, dcf=dcf, use_dcf=False,
        use_focal=True, focal_alpha=1.0, focal_normalize=True,
        focal_warmup_progress=0.0,
    )
    loss_plain = composable_kspace_loss(
        y_pred, y_target, dcf=dcf, use_dcf=False, use_focal=False,
    )
    assert torch.isclose(loss_warmup0, loss_plain, rtol=1e-5), \
        f"warmup_progress=0 should equal plain MSE: {float(loss_warmup0)} vs {float(loss_plain)}"
    print(f"PASS warmup_ramp: warmup=0 matches plain MSE ({float(loss_warmup0):.6e})")


def test_focal_log_matrix():
    """log compression keeps weights finite and bounded"""
    y_pred, y_target, dcf = _make_inputs()
    loss, diag = composable_kspace_loss(
        y_pred, y_target, dcf=dcf, use_dcf=False,
        use_focal=True, focal_alpha=1.0, focal_log_matrix=True,
        return_diagnostics=True,
    )
    assert torch.isfinite(loss), "log-matrix loss not finite"
    assert diag["w_focal_max"] < 100, f"log-matrix weight too large: {diag['w_focal_max']}"
    print(f"PASS focal_log_matrix: loss={float(loss):.6e}, w_max={diag['w_focal_max']:.3f}")


def test_diagnostics_keys():
    """diagnostics dict contains expected keys"""
    y_pred, y_target, dcf = _make_inputs()
    loss, diag = composable_kspace_loss(
        y_pred, y_target, dcf=dcf, use_dcf=True, dcf_power=1.0,
        use_focal=True, focal_alpha=1.0,
        return_diagnostics=True,
    )
    expected_keys = {"w_focal_mean", "w_focal_max", "w_focal_min", "w_focal_p99", "top1pct_loss_frac"}
    missing = expected_keys - set(diag.keys())
    assert not missing, f"missing diagnostic keys: {missing}"
    for k, v in diag.items():
        assert isinstance(v, float) and v == v, f"diag {k} = {v}"
    print(f"PASS diagnostics_keys: {sorted(diag.keys())}")


if __name__ == "__main__":
    tests = [
        test_gradient_isolation,
        test_toggle_independence_off,
        test_envelope_only_equivalence,
        test_complex_gradient_flow,
        test_warmup_ramp,
        test_focal_log_matrix,
        test_diagnostics_keys,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failures += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failures += 1
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(0 if failures == 0 else 1)
