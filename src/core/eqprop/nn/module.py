from abc import ABC, abstractmethod
from typing import Any, C

import torch
import torch.nn as nn

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
        return EqPropFunc.apply(self, x)

    @abstractmethod
    def _eqprop(self, x, dy):
        """Internelly called by EqPropFunc.backward()."""
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
        eqprop_layer._eqprop((input, grad_output))
        grad_input = None  # dL/dx = g*(dV_nudge -dV_free)/beta
        return grad_input


class EqPropLinear(EqPropBase, nn.Linear):
    """EqProp wrapper for nn.Linear."""

    def __init__(self, *args, **kwargs):
        super(self, EqPropBase).__init__(*args, **kwargs)
        super(self, nn.Linear).__init__(*args, **kwargs)

    def forward(self, x):
        return super(self, EqPropBase).forward(x)


class EqPropConv2d(EqPropBase, nn.Conv2d):
    def __init__(self, activation: callable, beta: float = 0.1, *args: Any, **kwargs: Any) -> None:
        super(
            self,
        ).__init__(*args, **kwargs)
        super(self, nn.Conv2d).__init__(*args, **kwargs)

    def forward(self, x):
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
