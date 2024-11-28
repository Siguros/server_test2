from typing import Any, Literal, Optional

from aihwkit.simulator.tiles.module import TileModule
from aihwkit.simulator.tiles.periphery import TileWithPeriphery
from aihwkit.simulator.tiles.base import SimulatorTile

NormType = Literal["nuc", "fro", "inf", "1", "-inf", "2"]  # codespell:ignore fro


class TileModuleWithPeriphery(TileModule, TileWithPeriphery):
    """Dummy logical tile class for type annotation purposes."""

    def __init__(self) -> None:
        self.tile: SimulatorTile  # physical tile

    ...
