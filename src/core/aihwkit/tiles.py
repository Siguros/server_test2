import torch.nn as nn
from aihwkit.nn.modules.base import AnalogLayerBase
from aihwkit.simulator.tiles.base import SimulatorTile
from aihwkit.simulator.tiles.periphery import TileWithPeriphery

__all__ = ["override_program_weights"]


class ProgramOverrideMeta(type):
    """Metaclass to override & rename the program_weights function of a tile."""

    _instances = {}

    def __new__(cls, name, bases, namespace, override_func):
        """Create a new class with the overridden program_weights function with singltone
        pattern."""
        new_cls_name = bases[0].__name__ + "_" + override_func.__name__
        if new_cls_name in cls._instances.keys():
            return cls._instances[new_cls_name]
        else:
            namespace["program_weights"] = override_func
            cls._instances[new_cls_name] = type.__new__(cls, new_cls_name, bases, namespace)
            return cls._instances[new_cls_name]


def new_atile_cls(tile_class: type[SimulatorTile], override_func: callable) -> type[SimulatorTile]:
    """Create a new tile class with an overridden program_weights function."""

    class OverridenAnalogTile(
        tile_class, metaclass=ProgramOverrideMeta, override_func=override_func
    ):
        pass

    return OverridenAnalogTile


# TODO: Utilize the new_atile_cls function to create a new tile class and replace the existing tile with the new one.


def _replace_atile(amodule: AnalogLayerBase, new_program_func):
    """Replace the program_weights function of a tile module."""
    for atile in amodule.analog_tiles():
        # AtileCls = new_tile_cls(atile.tile.__class__, new_program_func)
        # new_atile = AtileCls(atile.tile.device, atile.tile.in_features, atile.tile.out_features)
        atile.program_weights = new_program_func.__get__(atile, TileWithPeriphery)
        atile._get_name = (lambda self: f"Overridden{self.__class__.__name__}").__get__(
            atile, nn.Module
        )


def override_program_weights(model: nn.Module, new_program_func: callable):
    """Override the program_weights function of a tile module."""
    if hasattr(model, "analog_layers"):
        for amodule in model.analog_layers():
            _replace_atile(amodule, new_program_func)
    else:
        raise AttributeError(
            "Model does not have analog_layers attribute. Check if the model is a aihwkit model."
        )
