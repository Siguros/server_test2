import pytest
import torch


def test_const_idealized_rpu(idealized_analogtile) -> None:
    """Test whether the idealized rpu configuration behaves ideal."""
    atile = idealized_analogtile
    prev_weights = atile.tile.get_weights().flatten().numpy()
    x = torch.rand(1) * 0.1
    e = torch.rand(1) * 0.1
    atile.tile.update(x, e, False)
    current_weights = atile.tile.get_weights().flatten().numpy()
    assert current_weights - prev_weights == pytest.approx(-x * e, abs=1e-5)
