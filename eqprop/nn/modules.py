from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class EqProp(ABC):
    """EqProp base class.

    Args:
        ABC (_type_): _description_
    """

    def __init__(self, beta=0.1, *args, **kwargs):
        self.iseqprop = True
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


class EqPropMixin_(torch.autograd.Function):
    """Wrapper for EqProp that has same interface as nn.Module Full control over eqprop process
    (data)"""

    @staticmethod
    def forward(ctx, eqprop_layer: EqProp, input):
        ctx.eqprop_layer = eqprop_layer
        output = eqprop_layer.forward(input)
        # ctx.save_for_backward(input, output) # output is not needed
        ctx.mark_non_differentiable(output)
        ctx.save_for_backward(input)
        return output

    # : implement backward explicitly
    @staticmethod
    # @torch.once_differentiable()
    def backward(ctx, grad_output):
        # input, output = ctx.saved_tensors
        input = ctx.saved_tensors
        eqprop_layer: EqProp = ctx.eqprop_layer
        eqprop_layer.eqprop((input, grad_output))
        grad_input = None  # dummy, dy/dx?
        return grad_input


class EqPropLinear(EqProp, nn.Linear):
    """EqProp wrapper for nn.Linear."""

    def __init__(self, *args, **kwargs):
        super(self, nn.Linear).__init__(*args, **kwargs)
        if self.iseqprop:
            self = EqPropMixin_(self)

    def forward(self, x):
        return super().forward(x)


class EqPropSequential(EqProp, nn.Sequential):
    """EqProp wrapper for nn.Sequential."""

    def __init__(self, *args, **kwargs):
        super(self, nn.Sequential).__init__(*args, **kwargs)
        for module in self:
            if module.iseqprop:
                module = EqPropMixin_(module)

    def forward(self, x):
        for module in self:
            x = module(x)
        return x
