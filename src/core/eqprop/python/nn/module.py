from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from hydra_zen import instantiate

from src.core.eqprop.python.solver import EqPropSolver
from src.utils import eqprop_utils

__all__ = ["EqPropLinear", "EqPropConv2d", "EqPropSequential", "to_eqprop"]


class PositiveEqPropFunc(torch.autograd.Function):
    """EqProp function class.

    This class behaves similar to activation functions in torch.nn.functionals. Determines specific
    EqProp implementation. e.g. 3rd order, 2nd order, etc. Used internally with EqPropMixin.
    """

    @staticmethod
    def forward(ctx, eqprop_layer: _EqPropMixin, input) -> torch.Tensor:
        """Free phase for EqProp."""
        ctx.eqprop_layer = eqprop_layer
        positive_node = eqprop_layer.solver(input)
        ctx.save_for_backward(input, positive_node)
        return positive_node

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output) -> torch.Tensor:
        """Backward pass for EqProp."""
        input, positive_node = ctx.saved_tensors
        eqprop_layer: _EqPropMixin = ctx.eqprop_layer
        negative_node = eqprop_layer.solver(input, grad=grad_output)
        nodes = (positive_node, negative_node)
        eqprop_layer.calc_n_set_param_grad_(input, nodes)
        grad_input = eqprop_layer.calc_x_grad(nodes)  # dL/dx = g*(dV_nudge -dV_free)/beta
        return None, grad_input


class AlteredEqPropFunc(PositiveEqPropFunc):
    """Flip beta for every nudge phase."""

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output) -> torch.Tensor:
        """Backward pass for AlteredEqProp."""
        input, positive_node = ctx.saved_tensors
        eqprop_layer: _EqPropMixin = ctx.eqprop_layer
        eqprop_layer.solver.flip_beta()
        negative_node = eqprop_layer.solver(input, grad=grad_output)
        nodes = (positive_node, negative_node)
        eqprop_layer.calc_n_set_param_grad_(input, nodes)
        grad_input = eqprop_layer.calc_x_grad(nodes)
        return None, grad_input


class CenteredEqPropFunc(PositiveEqPropFunc):
    """Centered EqProp.

    Use 2 opposite nudged phases to calculate gradient.
    """

    @staticmethod
    def forward(ctx, eqprop_layer: _EqPropMixin, input):
        """Forward pass for centered EqProp."""
        ctx.eqprop_layer = eqprop_layer
        free_node = eqprop_layer.solver(input)
        # ctx.mark_non_differentiable(free_node)
        ctx.save_for_backward(input)
        return free_node

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """Backward pass for centered EqProp."""
        input = ctx.saved_tensors
        eqprop_layer: _EqPropMixin = ctx.eqprop_layer
        positive_node = eqprop_layer.solver(input, grad=grad_output)
        eqprop_layer.solver.flip_beta()
        negative_node = eqprop_layer.solver(input, grad=grad_output)
        nodes = (positive_node, negative_node)
        eqprop_layer.calc_n_set_param_grad_(input, nodes)
        grad_input = eqprop_layer.calc_x_grad(nodes)
        return None, grad_input


class _EqPropMixin(ABC):
    """EqProp mixin class.

    Wraps EqProp function and solver for nn.Module.

    Args:
        solver (EqPropSolver): EqProp solver
        eqprop_fn (PositiveEqPropFunc): EqProp wrapper function
    """

    IS_CONTAINER: bool = False

    def __init__(
        self,
        solver: EqPropSolver | None,
        eqprop_fn: torch.autograd.Function = PositiveEqPropFunc,
    ) -> None:
        self.eqprop_fn = eqprop_fn.apply
        if solver is None:
            # move import under here to avoid circular import
            from configs.eqprop.solver import (
                AnalogEqPropSolverConfig,
                IdealRectifierConfig,
                ProxQPStrategyConfig,
            )

            rectifier_cfg = IdealRectifierConfig(Vl=0.1, Vr=0.9)
            strategy_cfg = ProxQPStrategyConfig(amp_factor=1.0, activation=rectifier_cfg)
            cfg = AnalogEqPropSolverConfig(beta=0.1, strategy=strategy_cfg)
            solver = instantiate(cfg)
        self.solver = solver
        self.solver.set_model(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for EqProp."""
        x.requires_grad_()
        return self.eqprop_fn(self, x)

    @abstractmethod
    def calc_n_set_param_grad_(
        self, x: torch.Tensor, nodes: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Calculate & set gradient in-place for params.

        This function is called by EqPropFunc.backward.
        """

    @abstractmethod
    def calc_x_grad(self, nodes: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculate & return gradient for input dy/dx.

        This function is called by EqPropFunc.backward.
        """


class EqPropLinear(_EqPropMixin, nn.Linear):
    """EqProp wrapper for nn.Linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        eqprop_fn: torch.autograd.Function = PositiveEqPropFunc,
        solver: EqPropSolver | None = None,
        param_init_args: dict = {"min_w": 1e-6, "max_w": None, "max_w_gain": 0.28},
    ):
        """Initialize EqPropLinear.

        Args:
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            bias (bool, optional): If set to ``False``, the layer will not learn an additive bias. Defaults to ``True``.
            eqprop_fn (torch.autograd.Function, optional): specific Eqprop algorithm. Defaults to PositiveEqPropFunc.
            solver (Optional[EqPropSolver], optional): _description_. Defaults to None.
            param_init_args (dict, optional): args for set positive param init.
                Defaults to {"min_w":1e-6, "max_w":None, "max_w_gain":0.28}.
        """
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)
        _EqPropMixin.__init__(self, solver, eqprop_fn)
        self.apply(eqprop_utils.positive_param_init(**param_init_args))

    def forward(self, x):
        """Forward pass for EqPropLinear."""
        return _EqPropMixin.forward(self, x)

    @torch.no_grad()
    def calc_n_set_param_grad_(self, x: torch.Tensor, nodes: tuple[torch.Tensor, torch.Tensor]):
        """Calculate & set gradients of parameters manually.

        dy/dw = (nudge_dV^2 - free_dV^2)/beta \n
        = [prev_negative^2 + n_node^2 \n
        \\- (prev_positive^2 + p_node^2) \n
        \\- 2(prev_negative.T@n_node - prev_positive@p_node)]/beta \n
        = [2(p_node@x - n_node@x) \n
        \\+ 0 \n
        \\+ n_node^2 - p_node^2]/beta

        Args:
            x (torch.Tensor): input tensor
            nodes (tuple[torch.Tensor, torch.Tensor]): positive and negative nodes
        """
        beta = self.solver.beta
        positive_node, negative_node = nodes
        # Weight gradient
        dw = 2 * (
            torch.bmm(positive_node.unsqueeze(2), x.unsqueeze(1)).mean(0)
            - torch.bmm(negative_node.unsqueeze(2), x.unsqueeze(1)).mean(0)
        )
        # x.pow(2) - x.pow(2) = 0
        dw += (negative_node.pow(2).mean(0) - positive_node.pow(2).mean(0)).unsqueeze(1)
        dw /= beta
        self.weight.grad = dw if self.weight.grad is None else self.weight.grad + dw

        # Bias gradient
        if self.bias is not None:
            db = ((negative_node - positive_node) * (negative_node + positive_node - 2)).mean(
                0
            ) / beta
            self.bias.grad = db if self.bias.grad is None else self.bias.grad + db

    @torch.no_grad()
    def calc_x_grad(self, nodes: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculate & return gradient for input dy/dx. dy/dx = g*(dV_nudge -dV_free)/beta.

        Args:
            nodes (tuple[torch.Tensor, torch.Tensor]): positive and negative nodes
            dy (torch.Tensor): gradient of output
        """
        positive_node, negative_node = nodes
        return ((positive_node - negative_node) / self.solver.beta) @ self.weight


class EqPropConv2d(_EqPropMixin, nn.LazyConv2d):

    def __init__(
        self,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...] = (1, 1),
        padding: tuple[int, ...] | str = (0, 0),
        dilation: tuple[int, ...] = (1, 1),
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        eqprop_fn: torch.autograd.Function = PositiveEqPropFunc,
        device=None,
        dtype=None,
        solver: EqPropSolver | None = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        nn.LazyConv2d.__init__(
            self,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )
        _EqPropMixin.__init__(self, solver, eqprop_fn)
        self.unfold = nn.Unfold(
            kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride
        )

    def forward(self, x):
        """Forward pass for EqPropConv2d.

        Equivalent to Unfold input tensor and apply to EqPropLinear.
        """
        self.initialize_parameters(x)
        x = self.unfold(x).transpose(1, 2)
        x = _EqPropMixin.forward(self, x)
        batch, out_channels = x.shape
        x = x.view(batch, out_channels, *self.output_shape)
        return x

    @torch.no_grad()
    def calc_x_grad(self, nodes, dy):
        raise NotImplementedError("calculate_x_grad not implemented")


class EqPropSequential(_EqPropMixin, nn.Sequential):
    """EqProp wrapper for nn.Sequential.

    Merges multiple EqProp submodules and solve equilibrium altogether. This is different from
    putting multiple EqProp layers into nn.Sequential.
    """

    IS_CONTAINER = True

    def __init__(self, *args, **kwargs):
        nn.Sequential.__init__(self, *args, **kwargs)
        # EqPropMixin.__init__(self, solver, eqprop_fn)
        self.check_all_eqprop()

    def forward(self, x):
        """Forward pass for EqPropSequential."""
        raise NotImplementedError("forward not implemented")

    def check_all_eqprop(self):
        """Check if all submodules are EqProp."""
        for module in self:
            if not isinstance(module, _EqPropMixin):
                raise ValueError("All submodules must be EqProp.")


# TODO: referencing huggingface's PEFT get_peft_model()
def to_eqprop(submodule: nn.Module):
    """Convert submodule into EqPropModule."""
    if isinstance(submodule, nn.Linear):
        return EqPropLinear(submodule)
    elif isinstance(submodule, nn.Conv2d):
        return EqPropConv2d(submodule)
    elif isinstance(submodule, nn.Sequential):
        return EqPropSequential(submodule)
    else:
        raise NotImplementedError(f"EqProp not implemented for {submodule}")
