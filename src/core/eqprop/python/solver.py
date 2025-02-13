from __future__ import annotations

from typing import Any, Optional
from collections.abc import Sequence

import torch
import torch.nn as nn

from src.core.eqprop.python.strategy import AbstractStrategy
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class EqPropSolver:
    """Solve for the equilibrium point of the network.

    Args:
        strategy (AbstractStrategy|str): strategy to solve for the equilibrium point of the network.
        activation (Callable): activation function.
        amp_factor (float, optional): inter-layer potential amplifying factor. Defaults to 1.0.
        beta (float, optional): nudging factor. Defaults to 0.0.
    """

    # # singletons
    # def __new__(cls, *args, **kwargs):
    #     if not hasattr(cls, "_instance"):
    #         cls._instance = super().__new__(cls)
    #     return cls._instance

    def __init__(
        self,
        amp_factor: float,
        beta: float,
        strategy: AbstractStrategy,
    ) -> None:
        self.amp_factor = amp_factor
        self.beta = beta
        self.strategy = strategy

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value: float | int):
        self._beta = value

    def flip_beta(self) -> None:
        """Flip the sign of beta."""
        self.beta = -self.beta

    def set_model(self, model: torch.nn.Module) -> None:
        """Set strategy parameters from nn.Module."""
        st = self.strategy
        if not st.W and not st.B:
            for name, param in model.named_parameters():
                if name.endswith("weight"):
                    st.W.append(param)
                    st.dims.append(param.shape[0])
                elif name.endswith("bias"):
                    st.B.append(param)

    def __call__(
        self,
        x: torch.Tensor,
        grad: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Call the solver.

        Args:
            x (torch.Tensor): input of the network.
            nudge_phase (bool, optional): Defaults to False.
            return_energy (bool, optional): Defaults to False.
        """
        i_ext = None

        if grad is not None:
            i_ext = self.beta * grad
            log.debug(f"i_ext: {i_ext.abs().mean():.3e}")
        else:
            self.strategy.reset()
        nodes = self.strategy.solve(x, i_ext, **kwargs)
        return nodes

    def energy(self, Nodes, x) -> torch.Tensor:
        """Energy function."""
        it = len(Nodes)
        act = self.strategy.activation

        def layer_energy(n: torch.Tensor, w: nn.Module, m: torch.Tensor):
            """Energy function for a layer.

            Args:
                n (torch.Tensor):B x I
                w (torch.nn.Linear): O x I
                m (torch.Tensor): B x O

            Returns:
                E_layer = E_nodes - E_weights - E_biases
            """
            nodes_energy = 0.5 * torch.sum(torch.pow(n, 2), dim=1)
            weights_energy = 0.5 * (torch.matmul(act(m), w.weight) * act(n)).sum(dim=1)
            biases_energy = torch.matmul(act(m), w.bias) if getattr(w, "bias") is not None else 0.0
            return nodes_energy - weights_energy - biases_energy

        for idx in range(it):
            if idx == 0:
                E = layer_energy(x, self.W[idx], Nodes[idx])
            else:
                E += layer_energy(Nodes[idx - 1], self.W[idx], Nodes[idx])
        E += 0.5 * torch.sum(torch.pow(Nodes[-1], 2), dim=1)  # add E_nodes of output layer
        return E

    def total_energy(self, Nodes, x, y, beta) -> torch.Tensor:
        """Compute Total Free Energy: Wsum rho(u_i)W_{ij}rho(u_j)"""
        E = self.energy(Nodes, x)
        L = None
        if beta != 0:
            assert y is not None, ValueError("y must be provided if beta != 0")
            L = self.criterion(Nodes[-1], y)
            E += beta * L


class AnalogEqPropSolver(EqPropSolver):
    """Solver for analog resistive network with EqProp."""

    def __init__(
        self,
        amp_factor: float,
        beta: float,
        strategy: AbstractStrategy,
    ) -> None:
        super().__init__(amp_factor, beta, strategy)

    # TODO: Check validity when amp_factor is not 1
    def energy(self, Nodes, x) -> torch.Tensor:
        """Energy function."""
        if self.amp_factor != 1:
            raise NotImplementedError(
                "energy function for analog EqProp is not implemented when amp_factor != 1"
            )
        num_layers = len(Nodes)
        assert num_layers == len(self.dims) - 1, ValueError(
            "number of nodes must match the number of layers"
        )

        # TODO: Add bias
        def layer_power(submodule: nn.Module, in_V: torch.Tensor, out_V: torch.Tensor):
            r"""Energy function for a layer.

            Args:
                in_V (torch.Tensor):B x I
                submodule (torch.nn.Linear): O x I
                out_V (torch.Tensor): B x O

            Returns:
                E_layer = 0.5 * \Sum{G * (n_i - m_i)^2} : B

            """
            W = submodule.weight
            in_V = in_V.unsqueeze(1)
            out_V = out_V.unsqueeze(2)
            return (
                0.5 * torch.bmm(in_V.pow(2), W).sum(dim=(1, 2))
                + torch.bmm(W, out_V.pow(2)).sum(dim=(1, 2))
                - 2 * (in_V @ W @ out_V).squeeze()
            )

        for idx in range(num_layers):
            if idx == 0:
                E = layer_power(x, self.W[idx], Nodes[idx]) + self.activation.p(Nodes[idx])
            elif idx != num_layers - 1:
                E += layer_power(
                    self.amp_factor * (Nodes[idx - 1]), self.W[idx], Nodes[idx]
                ) + self.activation.p(Nodes[idx])
            else:
                E += layer_power(self.amp_factor(Nodes[idx - 1]), self.W[idx], Nodes[idx])
        return E
