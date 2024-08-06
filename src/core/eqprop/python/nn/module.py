from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
import torch.nn as nn

from src.core.eqprop.python.solver import EqPropSolver

__all__ = ["EqPropLinear", "EqPropConv2d", "EqPropSequential", "to_eqprop"]


class PositiveEqPropFunc(torch.autograd.Function):
    """EqProp function class.

    This class behaves similar to activation functions in torch.nn.functionals. Determines specific
    EqProp implementation. e.g. 3rd order, 2nd order, etc. Used internally with EqPropMixin.
    """

    @staticmethod
    def forward(ctx, eqprop_layer: EqPropMixin, input) -> torch.Tensor:
        """Free phase for EqProp."""
        ctx.eqprop_layer = eqprop_layer
        positive_node, _ = eqprop_layer.solver(input, nudge_phase=False)
        ctx.mark_non_differentiable(positive_node)
        ctx.save_for_backward(input, positive_node)
        return positive_node[-1]

    @staticmethod
    # @torch.once_differentiable()
    def backward(ctx, grad_output) -> torch.Tensor:
        """Backward pass for EqProp."""
        input, positive_node = ctx.saved_tensors
        eqprop_layer: EqPropMixin = ctx.eqprop_layer
        negative_node, _ = eqprop_layer.solver(input, nudge_phase=True, grad=grad_output)
        nodes = (positive_node, negative_node)
        eqprop_layer.calc_n_set_param_grad_(input, nodes)
        grad_input = eqprop_layer.calc_x_grad(nodes)  # dL/dx = g*(dV_nudge -dV_free)/beta
        return grad_input


class AlteredEqPropFunc(PositiveEqPropFunc):
    """Flip beta for every nudge phase."""

    @staticmethod
    def backward(ctx, grad_output) -> torch.Tensor:
        """Backward pass for AlteredEqProp."""
        input, positive_node = ctx.saved_tensors
        eqprop_layer: EqPropMixin = ctx.eqprop_layer
        eqprop_layer.solver.flip_beta()
        negative_node = eqprop_layer.solver(input, nudge_phase=True, grad=grad_output)
        nodes = (positive_node, negative_node)
        eqprop_layer.calc_n_set_param_grad_(input, nodes)
        grad_input = eqprop_layer.calc_x_grad(nodes)  # dL/dx = g*(dV_nudge -dV_free)/beta
        return grad_input


class CenteredEqPropFunc(PositiveEqPropFunc):
    """Centered EqProp.

    Use 2 opposite nudged phases to calculate gradient.
    """

    @staticmethod
    def forward(ctx, eqprop_layer: EqPropMixin, input):
        """Forward pass for centered EqProp."""
        ctx.eqprop_layer = eqprop_layer
        free_node = eqprop_layer.solver(input)
        ctx.mark_non_differentiable(free_node)
        ctx.save_for_backward(input)
        return free_node

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for centered EqProp."""
        input = ctx.saved_tensors
        eqprop_layer: EqPropMixin = ctx.eqprop_layer
        positive_node = eqprop_layer.solver(input, nudge_phase=True, grad=grad_output)
        eqprop_layer.solver.flip_beta()
        negative_node = eqprop_layer.solver(input, nudge_phase=True, grad=grad_output)
        nodes = (positive_node, negative_node)
        eqprop_layer.calc_n_set_param_grad_(input, nodes)
        grad_input = eqprop_layer.calc_x_grad(nodes)
        return grad_input


class EqPropMixin(ABC):
    """EqProp mixin class.

    Wraps EqProp function and solver for nn.Module.

    Args:
        solver (EqPropSolver): EqProp solver
        eqprop_fn (PositiveEqPropFunc): EqProp wrapper function
    """

    IS_CONTAINER: bool = False

    def __init__(
        self, solver: EqPropSolver, eqprop_fn: torch.autograd.Function = PositiveEqPropFunc
    ) -> None:
        self.eqprop_fn = eqprop_fn.apply
        self.solver = solver
        self.solver.set_model(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for EqProp."""
        return self.eqprop_fn(self, x)

    @abstractmethod
    def calc_n_set_param_grad_(
        self, x: torch.Tensor, nodes: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Calculate & set gradient in-place for params."""

    @abstractmethod
    def calc_x_grad(self, nodes: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculate & return gradient for input dy/dx."""


class EqPropLinear(EqPropMixin, nn.Linear):
    """EqProp wrapper for nn.Linear."""

    def __init__(
        self,
        solver: EqPropSolver,
        in_features: int,
        out_features: int,
        eqprop_fn: torch.autograd.Function = PositiveEqPropFunc,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, device=device, dtype=dtype)
        EqPropMixin.__init__(self, solver, eqprop_fn)

    def forward(self, x):
        """Forward pass for EqPropLinear."""
        return EqPropMixin.forward(self, x)

    @torch.no_grad()
    def calc_n_set_param_grad_(self, x: torch.Tensor, nodes: tuple[torch.Tensor, torch.Tensor]):
        """Calculate & set gradients of parameters manually.

        dy/dw = (nudge_dV^2 - free_dV^2)/beta \n
        = [prev_negative^2 - n_node^2 \n
        \\+ prev_positive^2 - p_node^2 \n
        \\- 2(prev_negative.T@n_node - prev_positive@p_node)]/beta

        Args:
            x (torch.Tensor): input tensor
            nodes (tuple[torch.Tensor, torch.Tensor]): positive and negative nodes
        """
        beta = self.solver.beta
        positive_node, negative_node = nodes
        # Weight gradient
        dw = 2 * (
            torch.bmm(negative_node.unsqueeze(2), x.unsqueeze(1)).mean(0)
            - torch.bmm(positive_node.unsqueeze(2), x.unsqueeze(1)).mean(0)
        )
        dw += (negative_node.pow(2).mean(0) - positive_node.pow(2).mean(0)).unsqueeze(1)
        dw /= beta
        self.weight.grad = dw if self.weight.grad is None else self.weight.grad + dw

        # Bias gradient
        if self.bias is not None:
            db = (
                2
                * [(negative_node - positive_node) * (negative_node + positive_node - 2)].mean(0)
                / beta
            )
            self.bias.grad = db if self.bias.grad is None else self.bias.grad + db

        # if hasattr(self, "weight"):
        #     if self.weight.grad is None:
        #         self.weight.grad = torch.zeros_like(self.weight)
        #     positive_node, negative_node = nodes
        #     prev_positive, prev_negative = x
        #     res = 2 * (
        #         torch.bmm(positive_node.unsqueeze(2), prev_positive.unsqueeze(1))
        #         .reshape((positive_node.size(0), *self.weight.shape))
        #         .mean(dim=0)
        #         - torch.bmm(negative_node.unsqueeze(2), prev_negative.unsqueeze(1))
        #         .reshape((positive_node.size(0), *self.weight.shape))
        #         .mean(dim=0)
        #     )
        #     # broadcast to 2D
        #     res += prev_negative.pow(2).mean(dim=0) - prev_positive.pow(2).mean(dim=0)
        #     res += (negative_node.pow(2).mean(dim=0) - positive_node.pow(2).mean(dim=0)).unsqueeze(1)
        #     self.weight.grad += res / self.solver.beta
        #     if self.bias is not None:
        #         if self.bias.grad is None:
        #             self.bias.grad = torch.zeros_like(self.bias)
        #         self.bias.grad += (
        #             (
        #                 (negative_node - positive_node)
        #                 * (
        #                     negative_node + positive_node - 2 * torch.ones_like(positive_node)
        #                 )  # (n-1)^2-(f-1)^2=2(n-f)(n+f-2)
        #             ).mean(dim=0)
        #             * 2
        #             / self.solver.beta
        #         )

    @torch.no_grad()
    def calc_x_grad(self, nodes: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Calculate & return gradient for input dy/dx. dy/dx = g*(dV_nudge -dV_free)/beta.

        Args:
            nodes (tuple[torch.Tensor, torch.Tensor]): positive and negative nodes
            dy (torch.Tensor): gradient of output
        """
        positive_node, negative_node = nodes
        return self.weight @ (positive_node - negative_node) / self.solver.beta


class EqPropConv2d(EqPropMixin, nn.Conv2d):

    def __init__(
        self,
        solver: EqPropSolver,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...],
        stride: tuple[int, ...],
        padding: tuple[int, ...],
        dilation: tuple[int, ...],
        transposed: bool,
        output_padding: tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        eqprop_fn: torch.autograd.Function = PositiveEqPropFunc,
        device=None,
        dtype=None,
    ) -> None:
        nn.Linear.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        EqPropMixin.__init__(self, solver, eqprop_fn)

    def forward(self, x):
        """Forward pass for EqPropConv2d.

        Equivalent to Unfold input tensor and apply to EqPropLinear.
        """
        x = self.unfold(x).transpose(1, 2)
        x = EqPropMixin.forward(self, x)
        batch, out_channels, _ = x.shape
        x = x.view(batch, out_channels, *self.output_shape)
        return x

    @torch.no_grad()
    def calc_x_grad(self, nodes, dy):
        raise NotImplementedError("calculate_x_grad not implemented")


class EqPropSequential(EqPropMixin, nn.Sequential):
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
            if not isinstance(module, EqPropMixin):
                raise ValueError("All submodules must be EqProp.")


# TODO: referencing huggingface's PEFT implementation
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
