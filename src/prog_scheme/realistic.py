from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Type

import torch
from aihwkit.exceptions import AnalogBiasConfigError, TileError, TorchTileConfigError
from aihwkit.simulator.configs import TorchInferenceRPUConfig
from aihwkit.simulator.parameters.enums import WeightClipType
from aihwkit.simulator.parameters.inference import WeightClipParameter, WeightModifierParameter
from aihwkit.simulator.tiles.base import SimulatorTile, SimulatorTileWrapper
from aihwkit.simulator.tiles.custom import CustomRPUConfig, CustomSimulatorTile
from aihwkit.simulator.tiles.functions import AnalogFunction
from aihwkit.simulator.tiles.inference import InferenceTileWithPeriphery
from aihwkit.simulator.tiles.module import TileModule

# TODO: 아래 메서드들 구현 필요
# aihwkit의 PulseType(aihwkit.simulator.parameters.enums.PulseType)들과 연동해서 구현.
# PCM inference noise 모델 등등 추가해보기

# Note: SimulaterTile.forward() 말고 AnalogNVM의 matmul에다가 구현해도 됨
# AnalogMVM.matmul()
# -> SimulatorTile.forward()
# -> PeripheryTile.joint_forward()
# -> AnalogFunction.forward()
# -> CustomTile.forward()
# -> PeripheryTile.read_weights()/program_weights()
# -> PeripheryTile.get_weights/set_weights(realistic=True) 에 쓰임


class _HalfSelectMixin:
    """Implements the half-selected update method."""

    def half_selection(self):
        pass


class HalfSelectedSimulatorTile(_HalfSelectMixin, CustomSimulatorTile):

    def set_weights(self, weight: torch.Tensor, **kwargs) -> None:
        """Set the tile weights.

        Args:
            weight: ``[out_size, in_size]`` weight matrix.
        """
        super().set_weights(weight, **kwargs)

    def get_weights(self) -> torch.Tensor:
        """Get the tile weights.

        Returns:
            a tuple where the first item is the ``[out_size, in_size]`` weight
            matrix; and the second item is either the ``[out_size]`` bias vector
            or ``None`` if the tile is set not to use bias.
        """
        super().get_weights()

    def update(
        self,
        x_input: torch.Tensor,
        d_input: torch.Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        """Implements rank-1 tile update with gradient noise (e.g. using pulse trains).

        Note:
            Ignores additional arguments

        Raises:
            TileError: in case transposed input / output or bias is requested
        """
        super().update(x_input, d_input, bias, in_trans, out_trans, non_blocking)
        self.half_selection()

    def forward(
        self,
        x_input: torch.Tensor,
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
        is_test: bool = False,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        if not is_test:
            noisy_weights = HalfSelectedSimulatorTile.modify_weight(
                self.weight, self._modifier, x_input.shape[0]
            )
        else:
            noisy_weights = self.weight

        ...

    @staticmethod
    def modify_weight(
        inp_weight: torch.Tensor, modifier: WeightModifierParameter, batch_size: int
    ):
        pass

    def set_config(self, rpu_config: "TorchInferenceRPUConfig") -> None:
        """Updated the configuration to allow on-the-fly changes.

        Args:
            rpu_config: configuration to use in the next forward passes.
        """
        self._f_io = rpu_config.forward
        self._modifier = rpu_config.modifier

    @torch.no_grad()
    def clip_weights(self, clip: WeightClipParameter) -> None:
        """Clip the weights. Called by InferenceTileWithperiphery.post_update_step()

        Args:
            clip: parameters specifying the clipping methof and type.

        Raises:
            NotImplementedError: For unsupported WeightClipTypes
            ConfigError: If unknown WeightClipType used.
        """

        if clip.type == WeightClipType.FIXED_VALUE:
            self.weight.data = torch.clamp(self.weight, -clip.fixed_value, clip.fixed_value)
        elif clip.type == WeightClipType.LAYER_GAUSSIAN:
            alpha = self.weight.std() * clip.sigma
            if clip.fixed_value > 0:
                alpha = min(clip.fixed_value, alpha)
            self.weight.data = torch.clamp(self.weight, -alpha, alpha)

        elif clip.type == WeightClipType.AVERAGE_CHANNEL_MAX:
            raise NotImplementedError
        else:
            raise TorchTileConfigError(f"Unknown clip type {clip.type}")


class RealisticTile(TileModule, InferenceTileWithPeriphery, SimulatorTileWrapper):
    """_summary_

    Note) methods in below are from CustomTile
    """

    def __init__(
        self,
        out_size: int,
        in_size: int,
        rpu_config: Optional["RPUConfigwithProgram"],  # type: ignore
        bias: bool = False,
        in_trans: bool = False,
        out_trans: bool = False,
    ):
        if in_trans or out_trans:
            raise TileError("in/out trans is not supported.")

        if not rpu_config:
            rpu_config = CustomRPUConfig()

        TileModule.__init__(self)
        SimulatorTileWrapper.__init__(
            self,
            out_size,
            in_size,
            rpu_config,  # type: ignore
            bias,
            in_trans,
            out_trans,
            torch_update=True,
        )
        InferenceTileWithPeriphery.__init__(self)

        if self.analog_bias:
            raise AnalogBiasConfigError("Analog bias is not supported for the torch tile")
        # dynamically add the program_weights method
        self.program_weights = rpu_config.program_weights.__get__(self, RealisticTile)

    def _create_simulator_tile(  # type: ignore
        self, x_size: int, d_size: int, rpu_config: "CustomRPUConfig"
    ) -> "SimulatorTile":
        """Create a simulator tile.

        Args:
            weight: 2D weight
            rpu_config: resistive processing unit configuration

        Returns:
            a simulator tile based on the specified configuration.
        """
        return rpu_config.simulator_tile_class(x_size=x_size, d_size=d_size, rpu_config=rpu_config)

    def forward(
        self, x_input: torch.Tensor, tensor_view: Optional[Tuple] = None  # type: ignore
    ) -> torch.Tensor:
        """Torch forward function that calls the analog context forward."""
        # pylint: disable=arguments-differ

        # to enable on-the-fly changes. However, with caution: might
        # change rpu config for backward / update while doing another forward.
        self.tile.set_config(self.rpu_config)

        out = AnalogFunction.apply(
            self.get_analog_ctx(), self, x_input, self.shared_weights, not self.training
        )

        if tensor_view is None:
            tensor_view = self.get_tensor_view(out.dim())
        out = self.apply_out_scaling(out, tensor_view)

        if self.digital_bias:
            return out + self.bias.view(*tensor_view)
        return out


@dataclass
class RPUConfigwithProgram(CustomRPUConfig):
    """Custom single RPU configuration."""

    program_weights: Callable[[Any], None] = None
    """Method to program the weights."""

    tile_class: Type = RealisticTile
    """Tile class that corresponds to this RPUConfig."""

    simulator_tile_class: Type = HalfSelectedSimulatorTile
    """Simulator tile class implementing the analog forward / backward / update."""

    clip: WeightClipParameter = field(
        default_factory=lambda: WeightClipParameter(
            type=WeightClipType.FIXED_VALUE, fixed_value=1.0
        )
    )
    modifier: WeightModifierParameter = field(default_factory=WeightModifierParameter)
