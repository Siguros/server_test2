from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from src.eqprop import eqprop_util


class EqProp(ABC):
    """EqProp base class.

    Args:
        activation (torch.nn.functional): activation function
        beta (float, optional): numerical approximation scale. Defaults to 0.1.
    """

    def __init__(
        self, activation: nn.functional.relu | eqprop_util.OTS, beta=0.1, *args: Any, **kwargs: Any
    ) -> None:
        self.iseqprop = True
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
    def forward(ctx, eqprop_layer: EqProp, input):
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
        eqprop_layer: EqProp = ctx.eqprop_layer
        eqprop_layer.eqprop((input, grad_output))
        grad_input = None  # dL/dx = g*(dV_nudge -dV_free)/beta
        return grad_input


class EqPropLinear(EqProp, nn.Linear):
    """EqProp wrapper for nn.Linear."""

    def __init__(self, *args, **kwargs):
        super(self, nn.Linear).__init__(*args, **kwargs)
        super(self, EqProp).__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x)


class EqPropSequential(EqProp, nn.Sequential):
    """EqProp wrapper for nn.Sequential."""

    def __init__(self, *args, **kwargs):
        super(self, nn.Sequential).__init__(*args, **kwargs)
        for module in self:
            if module.iseqprop:
                module = EqPropFunc(module)

    def forward(self, x):
        for module in self:
            x = module(x)
        return x
