import pytest
import torch

from src.core.aihwkit.utils import get_persistent_weights


def test_idealized_rpu_update(idealized_analogtile) -> None:
    """Test whether the idealized rpu configuration behaves ideal."""
    atile = idealized_analogtile
    prev_weights = atile.tile.get_weights()
    # since #max pulse \approx 2*10^3, and dw_min = 10^-5, max dw = 2*10^-2 thus we multiply by 0.1
    x = torch.rand(2) * 0.1
    e = torch.rand(2) * 0.1
    atile.tile.update(x, e, False)
    current_weights = atile.tile.get_weights()
    assert current_weights - prev_weights == pytest.approx(-torch.outer(e, x), abs=1e-5)


def test_idealized_rpu_read(idealized_analogtile) -> None:
    """Test whether the idealized rpu configuration reads the correct value."""
    atile = idealized_analogtile
    assert torch.all(get_persistent_weights(atile.tile) == atile.tile.get_weights())
