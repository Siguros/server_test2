from copy import deepcopy
from typing import Optional

import torch.nn as nn
from torch.nn import functional as F

from src.core.eqprop import nn as enn
from src.core.eqprop.python.solver import EqPropSolver
from src.utils import eqprop_utils


class MultiplyActivation(nn.Module):
    """Multiply Activation layer."""

    def __init__(self, scale: float = 1.0):
        """Initialize MultiplyActivation layer.

        Args:
            scale (float, optional): _description_. Defaults to 1.0.
        """
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
        beta: float = 0.01,
        bias: bool | list[bool] = [True, False],
        scale_input: int = 2,
        scale_output: int = 2,
        solver: Optional[EqPropSolver] = None,
        param_adjuster: Optional[eqprop_utils.AdjustParams] = eqprop_utils.AdjustParams(),
        dummy: bool = False,
        layer_scale: float = 4,
    ) -> None:
        """Initialize EqPropBackbone.

        Args:
            cfg (list[int], optional): Configuration of layers. Defaults to [784 * 2, 128, 10 * 2].
            bias (bool | list[bool], optional): Bias for each layer. Defaults to [True, False].
            scale_input (int, optional): Scale input. Defaults to 2.
            scale_output (int, optional): Scale output. Defaults to 2.
            solver (Optional[EqPropSolver], optional): Solver for EqProp. Defaults to None.
            param_adjuster (Optional[eqprop_utils.AdjustParams], optional): Parameter adjuster for every forward call.
                Defaults to eqprop_utils.AdjustParams().
            dummy (bool, optional): If true, Set all layers to nn.Linear with ReLU for testing. Defaults to False.
            layer_scale (float, optional): Scaling factor between eqprop layers. Defaults to 4.0.
        """
        super().__init__()
        layers = []
        for idx in range(len(cfg) - 1):
            bias_idx = bias if isinstance(bias, bool) else bias[idx]
            if dummy:  # for testing
                layers.append(nn.Linear(cfg[idx], cfg[idx + 1], bias=bias_idx))
                layers.append(nn.ReLU())
            else:
                solver_ = deepcopy(solver) if solver else None
                layers.append(
                    enn.EqPropLinear(cfg[idx], cfg[idx + 1], bias=bias_idx, solver=solver_)
                )
                # layers.append(nn.Tanh())
                layers.append(MultiplyActivation(scale=layer_scale))

        self.model = nn.Sequential(*layers)
        self.param_adjuster = param_adjuster
        eqprop_utils.interleave.set_num_input(scale_input)
        eqprop_utils.interleave.set_num_output(scale_output)

    @eqprop_utils.interleave(type="both")
    def forward(self, x):
        if self.param_adjuster is not None:
            self.model.apply(self.param_adjuster)
        return self.model(x)
