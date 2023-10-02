from abc import ABC, abstractmethod
from typing import Any, C

import torch
import torch.nn as nn

from src.eqprop import eqprop_util


class EqPropBase(ABC):
    """EqProp base class.

    Args:
        activation (torch.nn.functional): activation function
        beta (float, optional): numerical approximation scale. Defaults to 0.1.
    """

    IS_CONTAINER: bool = False

    def __init__(self, activation: callable, beta: float = 0.1, *args: Any, **kwargs: Any) -> None:
        self.activation = activation
        self.beta = beta

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def eqprop(self, x, dy):
        pass

    @abstractmethod
    def energy(self, x):
        pass


class EqPropFunc(torch.autograd.Function):
    """Wrapper for EqProp.

    Determines specific EqProp implementation. e.g. 3rd order, 2nd order, etc. Use with
    EqPropFunc.apply() instead of EqProp.forward().
    """

    @staticmethod
    def forward(ctx, eqprop_layer: EqPropBase, input):
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
        # input, output = ctx.saved_tensors
        input = ctx.saved_tensors
        eqprop_layer: EqPropBase = ctx.eqprop_layer
        eqprop_layer.eqprop((input, grad_output))
        grad_input = None  # dL/dx = g*(dV_nudge -dV_free)/beta
        return grad_input


class EqPropLinear(EqPropBase, nn.Linear):
    """EqProp wrapper for nn.Linear."""

    def __init__(self, *args, **kwargs):
        super(self, nn.Linear).__init__(*args, **kwargs)
        super(self, EqPropBase).__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x)


class EqPropSequential(EqPropBase, nn.Sequential):
    """EqProp wrapper for nn.Sequential."""

    def __init__(self, *args, **kwargs):
        super(self, nn.Sequential).__init__(*args, **kwargs)

    def forward(self, x):
        for module in self:
            x = module(x)
        return x
