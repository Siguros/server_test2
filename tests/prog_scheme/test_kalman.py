import pytest
import torch


def test_base_kf(ideal_analogtile, base_kf) -> None:
    """Test the result between the kalman filter estimation and the true value."""
    atile = ideal_analogtile
    curr_weights = atile.tile.get_weights()
    input_size, output_size = curr_weights.shape
    base_kf.x_est = curr_weights.clone().flatten().numpy()
    u = torch.randn(input_size) * 0.1
    v = torch.randn(output_size) * 0.1
    atile.tile.update(v, u, False)
    base_kf.predict(-torch.outer(u, v).flatten().numpy())
    assert atile.tile.get_weights().flatten().numpy() == pytest.approx(base_kf.x_est, abs=1e-5)


def test_linear_ekf(kf_analogtile, linear_ekf) -> None:
    """Test the result between the kalman filter estimation and the true value."""
    atile = kf_analogtile
    curr_weights = atile.tile.get_weights()
    input_size, output_size = curr_weights.shape
    # TODO: EKF 구현 순서 확인
    linear_ekf.x_est = curr_weights.clone().flatten().numpy()
    # linear_ekf.update(atile.target_weights.flatten().numpy())
    u = torch.randn(input_size) * 0.1
    v = torch.randn(output_size) * 0.1
    atile.tile.update(v, u, False)
    linear_ekf.predict(-torch.outer(u, v).flatten().numpy())
    assert atile.tile.get_weights().flatten().numpy() == pytest.approx(linear_ekf.x_est, abs=1e-3)
