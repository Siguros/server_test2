from enum import Enum

import torch
from aihwkit.simulator.configs.configs import MappableRPU
from aihwkit.simulator.parameters.helpers import _PrintableMixin
from aihwkit.simulator.tiles.base import SimulatorTile


def get_persistent_weights(tile: SimulatorTile) -> torch.Tensor:
    """Get the hidden noiseless weights from the physical tile.

    It is different from `tile.get_weights()` method in general.
    """
    name_list = tile.get_hidden_parameter_names()
    if "persistent_weights" in name_list:
        idx = name_list.index("persistent_weights")
        return tile.get_hidden_parameters()[idx]
    else:
        # Already noiseless
        return tile.get_weights()


def rpuconf2dict(rpuconfig: MappableRPU, max_depth=2, current_depth=0) -> dict:
    """Convert an RPUConfig object to a dictionary."""
    result = {}
    for key, val in rpuconfig.__dict__.items():
        if type(val) in (float, int, str, bool, type(None)):  # primitive
            result[key] = val
        elif isinstance(val, type):  # class
            result[key] = val.__name__
        elif isinstance(val, Enum):
            result[key] = val.name
        elif isinstance(val, _PrintableMixin):  # instance
            result[key] = (
                rpuconf2dict(val, max_depth, current_depth + 1)
                if current_depth < max_depth
                else str(val)
            )
            result[key]["is_default"] = True if val.__dict__ == val.__class__().__dict__ else False
        else:
            raise ValueError(f"Unknown type {type(val)} for {key}")
    return result
