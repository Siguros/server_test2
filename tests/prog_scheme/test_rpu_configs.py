import pytest
import torch


def test_const_idealized_rpu(idealized_analogtile) -> None:
    """Test whether the idealized rpu configuration behaves ideal."""
    atile = idealized_analogtile
    prev_weights = atile.tile.get_weights()
    # since #max pulse \approx 2*10^3, and dw_min = 10^-5, max dw = 2*10^-2 thus we multiply by 0.1
    x = torch.rand(2) * 0.1
    e = torch.rand(2) * 0.1
    atile.tile.update(x, e, False)
    current_weights = atile.tile.get_weights()
    assert current_weights - prev_weights == pytest.approx(-torch.outer(e, x), abs=1e-5)
