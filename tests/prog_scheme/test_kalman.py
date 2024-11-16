import pytest
import torch

from src.core.aihwkit.utils import get_persistent_weights


def test_base_kf(ideal_analogtile, base_kf) -> None:
    """Test the result between the kalman filter estimation and the true value."""
    tile = ideal_analogtile.tile
    curr_weights = get_persistent_weights(tile)
    input_size, output_size = curr_weights.shape
    base_kf.x_est = curr_weights.clone().flatten().numpy()
    u = torch.randn(input_size) * 0.1
    v = torch.randn(output_size) * 0.1
    tile.update(v, u, False)
    base_kf.predict(-torch.outer(u, v).flatten().numpy())
    assert get_persistent_weights(tile).flatten().numpy() == pytest.approx(base_kf.x_est, abs=1e-5)


def test_linear_ekf(kf_analogtile, linear_ekf) -> None:
    """Test the result between the kalman filter estimation and the true value."""
    tile = kf_analogtile.tile
    curr_weights = get_persistent_weights(tile)
    input_size, output_size = curr_weights.shape
    # TODO: EKF 구현 순서 확인
    linear_ekf.x_est = curr_weights.clone().flatten().numpy()
    # linear_ekf.update(atile.target_weights.flatten().numpy())
    u = torch.randn(input_size) * 0.1
    v = torch.randn(output_size) * 0.1
    tile.update(v, u, False)
    linear_ekf.predict(-torch.outer(u, v).flatten().numpy())
    assert get_persistent_weights(tile).flatten().numpy() == pytest.approx(
        linear_ekf.x_est, abs=5e-4
    )


def test_linear_ekf_jacobian(linear_ekf) -> None:
    """Test if the jacobian is correctly implemented."""
    dim = linear_ekf.dim
    x = torch.randn(dim).numpy()
    u = torch.randn(dim).numpy()
    dx = du = (linear_ekf._scale_up + linear_ekf._scale_down) / 2
    linear_ekf.x_est = x
    x_new = linear_ekf.f(x + dx, u + du)
    x_new_taylor = (
        linear_ekf.f(x, u)
        + linear_ekf.f_jacobian_x(x, u) @ dx
        + linear_ekf.f_jacobian_u(x, u) @ du
    )
    assert x_new == pytest.approx(x_new_taylor, abs=1e-3)
