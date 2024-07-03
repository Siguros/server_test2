from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from src.core.eqprop.solver import AnalogEqPropSolver, EqPropSolver


class EqPropBase(ABC):
    """EqProp base class.

    Args:
        activation (torch.nn.functional): activation function
        beta (float, optional): perturbation scale. Defaults to 0.1.
    """

    IS_CONTAINER: bool = False

    def __init__(
        self, activation: callable, beta: float = 0.1, solver: EqPropSolver = AnalogEqPropSolver()
    ) -> None:
        self.activation = activation
        self.beta = beta
        self.solver = solver
        self._init_nodes()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Forward pass for EqProp layer."""
        return EqPropFunc.apply(self, x)

    @abstractmethod
    def _eqprop(self, x: torch.Tensor, dy: torch.Tensor):
        """Internelly called by EqPropFunc.backward()."""
        pass

    @abstractmethod
    def _init_nodes(self, x: torch.Tensor):
        """Initialize positive & negative nodes."""
        pass


class EqPropFunc(torch.autograd.Function):
    """Wrapper for EqProp.

    Determines specific EqProp implementation. e.g. 3rd order, 2nd order, etc. Use with
    EqPropFunc.apply() instead of EqProp.forward().
    """

    @staticmethod
    def forward(ctx, eqprop_layer: EqPropBase, input):
        """Forward pass for EqProp."""
        ctx.eqprop_layer = eqprop_layer
        output = eqprop_layer.forward(input)
        # ctx.save_for_backward(input, output) # output is not needed
        ctx.mark_non_differentiable(output)
        ctx.save_for_backward(input)
        return output

    # : implement explicit backward pass
    @staticmethod
    # @torch.once_differentiable()
    def backward(ctx, grad_output):
        """Backward pass for EqProp."""
        # input, output = ctx.saved_tensors
        input = ctx.saved_tensors
        eqprop_layer: EqPropBase = ctx.eqprop_layer
        eqprop_layer._eqprop((input, grad_output))
        grad_input = None  # dL/dx = g*(dV_nudge -dV_free)/beta
        return grad_input


class EqPropLinear(EqPropBase, nn.Linear):
    """EqProp wrapper for nn.Linear."""

    def __init__(self, *args, **kwargs):
        super(self, EqPropBase).__init__(*args, **kwargs)
        super(self, nn.Linear).__init__(*args, **kwargs)

    def forward(self, x):
        """Forward pass for EqPropLinear."""
        return super(self, EqPropBase).forward(x)


class EqPropConv2d(EqPropBase, nn.Conv2d):
    def __init__(self, activation: callable, beta: float = 0.1, *args: Any, **kwargs: Any) -> None:
        super(
            self,
        ).__init__(*args, **kwargs)
        super(self, nn.Conv2d).__init__(*args, **kwargs)

    def forward(self, x):
        """Forward pass for EqPropConv2d.

        Equivalent to Unfold input tensor and apply to EqPropLinear.
        """
        x = self.unfold(x).transpose(1, 2)
        x = super(self, EqPropBase).forward(x)
        batch, out_channels, _ = x.shape
        x = x.view(batch, out_channels, *self.output_shape)
        return x


class EqPropSequential(EqPropBase, nn.Sequential):
    """EqProp wrapper for nn.Sequential."""

    def __init__(self, *args, **kwargs):
        super(self, nn.Sequential).__init__(*args, **kwargs)

    def forward(self, x):
        for module in self:
            x = module(x)
        return x


def to_eqprop(submodule: nn.Module):
    """Convert submodule to EqPropModule."""
    if isinstance(submodule, nn.Linear):
        return EqPropLinear(submodule)
    elif isinstance(submodule, nn.Conv2d):
        return EqPropConv2d(submodule)
    elif isinstance(submodule, nn.Sequential):
        return EqPropSequential(submodule)
    else:
        raise NotImplementedError(f"EqProp not implemented for {submodule}")
