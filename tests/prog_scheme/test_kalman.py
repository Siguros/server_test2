import pytest
import torch


def test_linear_ekf(kf_analogtile, linear_ekf) -> None:
    """Test the result between the kalman filter estimation and the true value."""
    atile = kf_analogtile
    curr_weights = atile.tile.get_weights()
    output_size, input_size = curr_weights.shape
    # TODO: EKF 구현 순서 확인
    linear_ekf.x_est = curr_weights.clone().flatten().numpy()
    # linear_ekf.update(atile.target_weights.flatten().numpy())
    u = torch.randn(input_size)
    v = torch.randn(output_size)
    atile.tile.update(v, u, False)
    linear_ekf.predict(-torch.outer(v, u).flatten().numpy())
    assert atile.tile.get_weights().flatten().numpy() == pytest.approx(linear_ekf.x_est, abs=1e-3)
