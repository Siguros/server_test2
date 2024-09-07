from typing import Optional

import torch.nn as nn
from torch.nn import functional as F

from src.core.eqprop import nn as enn
from src.utils import eqprop_utils


class MultiplyActivation(nn.Module):

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * x


class EqPropBackbone(nn.Module):
    """EqPropBackbone with Default EqPropLinear layers.

    Each EqPropLinear layer uses PositiveEqPropFunc as eqprop function and ProxQPStrategy as
    solver.
    """

    def __init__(
        self,
        cfg: list[int] = [784 * 2, 128, 10 * 2],
        bias: bool | list[bool] = [True, False],
        scale_input: int = 2,
        scale_output: int = 2,
        param_adjuster: Optional[eqprop_utils.AdjustParams] = eqprop_utils.AdjustParams(),
        dummy: bool = False,
    ) -> None:
        super().__init__()
        layers = []
        for idx in range(len(cfg) - 1):
            bias_idx = bias if isinstance(bias, bool) else bias[idx]
            if dummy:  # for testing
                layers.append(nn.Linear(cfg[idx], cfg[idx + 1], bias=bias_idx))
                layers.append(nn.ReLU())
            else:
                layers.append(enn.EqPropLinear(cfg[idx], cfg[idx + 1], bias=bias_idx))
                layers.append(nn.Tanh())
                layers.append(MultiplyActivation(scale=2))

        self.model = nn.Sequential(*layers)
        self.param_adjuster = param_adjuster
        eqprop_utils.interleave.set_num_input(scale_input)
        eqprop_utils.interleave.set_num_output(scale_output)

    @eqprop_utils.interleave(type="both")
    def forward(self, x):
        if self.param_adjuster is not None:
            self.model.apply(self.param_adjuster)
        return self.model(x)
