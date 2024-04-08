from collections import OrderedDict
from typing import Any, Callable, Iterator, List, Mapping, Sequence, Tuple, Union, overload

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.eqprop import eqprop_util
from src.core.eqprop.solver import AnalogEqPropSolver


class EP(nn.Module):
    def __init__(
        self,
        batch_size: int,
        beta: float = 1e-2,
        dims: list = [784, 500, 10],
        iters: tuple = (20, 4),
        activation=torch.sigmoid,
        epsilon: float = 0.5,
        criterion=nn.MSELoss(reduction="none"),
        L: Sequence = None,
        U: Sequence = None,
        *args,
        **kwargs,
    ):
        """Equilibrium Propagation (EP) model.

        Args:
            batch_size (_type_): _description_
            beta (float, optional): loss coupling strength. Defaults to 1e-2.
            dims (list, optional): dimensions of network architectures. Defaults to [784,500,10].
            iters (tuple, optional): Iterations for minimizing phases. Defaults to (200,4).
            activation (_type_, optional): nonlinear activation function for nodes. Defaults to torch.sigmoid.
            epsilon (float, optional): step size for minimizing Energy. Defaults to 0.5.
            criterion (_type_, optional): loss function. Defaults to nn.MSELoss(reduction='none').
        """

        super().__init__()

        self.batch_size = batch_size
        self.beta = beta
        self.bias: Union[bool, int] = kwargs.get("bias", True)
        self.pos_W = kwargs.get("pos_W", False)
        self.doubling = kwargs.get("doubling", False)
        self.num_classes = dims[-1]
        if self.doubling:
            # eqprop_util.interleave.on()
            self.num_classes //= 2
        self.eps = epsilon
        self.dims = dims
        self.L = L
        self.U = U
        assert len(iters) == 2, ValueError("2 iteration steps(free, nudge) required")
        self.free_iters, self.nudge_iters = iters
        self.activation = activation

        self._Nodes = [
            torch.empty((batch_size, n)).normal_(0.5, 0.5).clamp(0, 1).requires_grad_(True)
            for n in dims[1:]
        ]

        self.criterion = criterion
        # self.inner_optimizer = torch.optim.SGD(self._Nodes, lr=self.eps)
        self._init_weights()
        # self.metric_handler = metricHandler().setup(num_layers=len(self.dims) - 1)

    def _init_weights(self):
        # pos_W, bias
        self.W = nn.ModuleList()
        bias = False if self.bias == 2 else self.bias

        for idx in range(len(self.dims) - 1):
            if idx == 0 & self.bias == 2:
                self.W.append(nn.Linear(self.dims[idx], self.dims[idx + 1], bias=True))
                nn.init.constant_(self.W[idx].bias, 1)
            else:
                self.W.append(nn.Linear(self.dims[idx], self.dims[idx + 1], bias=bias))
            if self.pos_W:
                assert self.L is not None, ValueError("L is required for pos_W")
                assert self.U is not None, ValueError("U is required for pos_W")
                if isinstance(self.L, Union[float, int]):
                    self.L = [self.L] * (len(self.dims) - 1)
                    self.U = [self.U] * (len(self.dims) - 1)

                assert len(self.L) == len(self.dims) - 1, ValueError(
                    "L and U must have the same length as dims"
                )
                nn.init.uniform_(self.W[idx].weight, self.L[idx], self.U[idx])
            else:
                nn.init.xavier_uniform_(self.W[idx].weight)

    @eqprop_util.type_as
    def forward(self, x, y=None, beta=0.0) -> List[torch.Tensor]:
        """Relax Nodes till converge."""
        self.W.requires_grad_(False)  # freeze weights
        if beta == 0.0:
            opt_nodes = self.minimize(x, y, beta=beta, iters=self.free_iters)
        else:
            opt_nodes = self.minimize(x, y, beta=beta, iters=self.nudge_iters)
        return opt_nodes

    # TODO: implement a better algorithm to find the optimal nodes (e.g. Newton's method)

    def minimize(
        self,
        x,
        y=None,
        Nodes: List[torch.Tensor] = None,
        beta=0.0,
        iters=None,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Minimize the total energy function using torch.autograd."""
        Nodes = self._Nodes if Nodes is None else Nodes
        iters = self.free_iters if iters is None else iters
        self.W.requires_grad_(False)  # freeze weights
        for _ in range(iters):
            # print(Nodes)
            self.step(Nodes, x, y, beta)
        relaxedNodes1 = [nodes.clone().detach() for nodes in Nodes]
        return relaxedNodes1

    def step(self, Nodes: List[torch.Tensor], x, y=None, beta: float = 0.0) -> None:
        """Update Nodes one step.

        Args:
            Nodes (List[torch.Tensor]): _description_
            x (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            beta (float, optional): _description_. Defaults to 0.0.
        """
        # compute grads(dE/du)
        E, __ = self.Tenergy(Nodes, x, y, beta=beta)
        # update Nodes
        grads = torch.autograd.grad(E.sum(), Nodes)
        with torch.no_grad():
            for idx, layergrads in enumerate(grads):
                Nodes[idx] -= self.eps * layergrads
                Nodes[idx] = torch.clamp(Nodes[idx], 0, 1)
                Nodes[idx].requires_grad_(True)

    # TODO: make net._Nodes gradient zero after step
    def step_(self, Nodes: List[torch.Tensor], x, y=None, beta: float = 0.0) -> None:
        """Alternative method to step. Use autograd.backward() instead.

        Args:
            Nodes (List[torch.Tensor]): _description_
            x (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            beta (float, optional): _description_. Defaults to 0.0.
        """
        E, _ = self.Tenergy(Nodes, x, y, beta)
        E.sum().backward()
        with torch.no_grad():
            for idx, nodes in enumerate(Nodes):
                nodes -= self.eps * nodes.grad
                nodes = torch.clamp(nodes, 0, 1)
                nodes.requires_grad_(True)
                nodes.grad = None  # ...?

    def energy(self, Nodes: List[torch.Tensor], x) -> torch.Tensor:
        """Energy function."""
        it = len(Nodes)
        act = self.activation
        assert it == len(self.dims) - 1, ValueError(
            "number of nodes must match the number of layers"
        )
        assert it == len(self.W), ValueError("number of nodes must match the number of layers")

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

    def Tenergy(
        self, Nodes: List[torch.Tensor], x, y=None, beta: float = 0.0, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Total Free Energy: Wsum rho(u_i)W_{ij}rho(u_j)"""
        E = self.energy(Nodes, x)
        L = None
        if beta != 0:
            assert y is not None, ValueError("y must be provided if beta != 0")
            L = self.loss(Nodes[-1], y)
            E += beta * L
            L = L.mean().detach()
        return (E, L)

    @eqprop_util.interleave(type="in")
    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        if self.criterion.__class__.__name__.find("MSE") != -1:
            y = F.one_hot(y, num_classes=self.num_classes)
            L = self.criterion(y_hat.float(), y.float()).sum(dim=1).squeeze()
        else:
            L = self.criterion(y_hat.float(), y).squeeze()
        return L

    @torch.no_grad()
    def update(self, free_nodes: List[torch.Tensor], nudge_nodes: List[torch.Tensor], x) -> None:
        """Update weights with hardcoded gradients from theorm.

        dw_ij = (rho(un_i)rho(un_j) - rho(uf_i)rho(uf_j))/beta
        Annot.
          un<>: minimized nudge_nodes
          uf<>: minimized free_nodes

        Args:
            free_nodes (List[torch.Tensor]): _description_
            nudge_nodes (List[torch.Tensor]): _description_
            x (_type_): _description_
        """
        # lr = 1e-1
        act = self.activation
        free_nodes.insert(0, x)
        nudge_nodes.insert(0, x)
        self.W.requires_grad_(True)
        self.W.zero_grad()
        for idx, W in enumerate(self.W):
            W.weight.grad = (
                torch.matmul(
                    act(nudge_nodes[idx + 1]).mean(dim=0, keepdim=True).T,
                    act(nudge_nodes[idx]).mean(dim=0, keepdim=True),
                )
                - torch.matmul(
                    act(free_nodes[idx + 1]).mean(dim=0, keepdim=True).T,
                    act(free_nodes[idx]).mean(dim=0, keepdim=True),
                )
            ) / (-self.beta)
            # consider bias as synaptic weights with u_i = 1
            if getattr(W, "bias") is not None:
                W.bias.grad = (
                    act(nudge_nodes[idx + 1]).mean(dim=0) - act(free_nodes[idx + 1]).mean(dim=0)
                ) / (-self.beta)

    def update_(self, free_nodes, nudge_nodes, x, y) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Update weights from optimized free_nodes & nudge_nodes using autograd.backward()

        Args:
            free_nodes (_type_): _description_
            nudge_nodes (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Any]: Free Energy, Nudge Energy, (optional) loss
        """
        self.W.requires_grad_(True)
        self.W.zero_grad()
        # set W.grads
        Efs, _ = self.Tenergy(free_nodes, x, y, beta=0.0)
        Ef = Efs.mean()
        Ef.backward(retain_graph=True)
        Ens, loss = self.Tenergy(nudge_nodes, x, y, beta=self.beta)
        En = Ens.mean()
        (-En).backward()
        return (Ef.clone().detach(), En.clone().detach(), loss)

    @property
    def Nodes(self) -> List[torch.Tensor]:
        """Get nodes."""
        Nodes = [nodes.clone().detach().requires_grad_(True) for nodes in self._Nodes]
        return Nodes


class AnalogEP(EP):
    """Use slightly different energy(pseudopower)

    Attrs:
        _Nodes (List[torch.Tensor]): output node voltages of each layer
    """

    def __init__(
        self,
        batch_size: int,
        beta: float = 1e-2,
        dims: list = [784, 500, 10],
        activation=lambda x: x,
        epsilon: float = 0.2,
        criterion=nn.MSELoss(reduction="none"),
        *args,
        **kwargs,
    ):
        super().__init__(
            batch_size,
            beta,
            dims,
            (1, 1),
            activation,
            epsilon,
            criterion,
            *args,
            **kwargs,
        )
        DeprecationWarning("AnalogEP is deprecated. Use AnalogEP2 instead.")

    def _init_weights(self):
        super()._init_weights()

    def energy(self, Nodes, x) -> torch.Tensor:
        if not hasattr(self, "Is"):
            self.Is = 1e-6
            self.Vl = 0.1
            self.Vr = 0.9

        num_layers = len(Nodes)
        act = self.activation
        assert num_layers == len(self.dims) - 1, ValueError(
            "number of nodes must match the number of layers"
        )
        assert num_layers == len(self.W), ValueError(
            "number of nodes must match the number of layers"
        )

        def layer_power(n: torch.Tensor, w: nn.Module, m: torch.Tensor):
            r"""Energy function for a layer.

            Args:
                n (torch.Tensor):B x I
                w (torch.nn.Linear): O x I
                m (torch.Tensor): B x O

            Returns:
                E_layer = 0.5 * \Sum{G * (n_i - m_i)^2} : B
            """
            return 0.5 * torch.sum(w.weight * self.deltaV(n, m).pow(2), dim=(1, 2))

        def rectifier_power(x: torch.Tensor):  # , Is=1e-8, Vt1=0.1, Vt2=0.9, eta=1):
            def diode_power(V, Vt):
                return 0.026 * self.Is * (torch.exp((V - Vt) / (0.026)) - 1) - self.Is * (V - Vt)

            return torch.sum(diode_power(-x, -self.Vl) + diode_power(x, self.Vr), dim=-1)

        for idx in range(num_layers):
            if idx == 0:
                E = layer_power(x, self.W[idx], Nodes[idx]) + rectifier_power(Nodes[idx])
            elif idx != num_layers - 1:
                E += layer_power(act(Nodes[idx - 1]), self.W[idx], Nodes[idx]) + rectifier_power(
                    Nodes[idx]
                )
            else:
                E += layer_power(act(Nodes[idx - 1]), self.W[idx], Nodes[idx])
        return E

    @classmethod
    def deltaV(cls, n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Compute deltaV matrix from 2 node voltages.

        Args:
            n (torch.Tensor): (B x) I
            m (torch.Tensor): (B x) O

        Returns:
            torch.Tensor: (B x) O x I
        """
        if len(n.shape) == 2:
            assert n.shape[0] == m.shape[0], ValueError("n and m must have the same batch size")
            N = n.clone().unsqueeze(dim=-1).repeat(1, 1, m.shape[-1]).transpose(1, 2)
            M = m.clone().unsqueeze(dim=-1).repeat(1, 1, n.shape[-1])
        elif len(n.shape) == 1:
            N = n.clone().unsqueeze(dim=-1).repeat(1, m.shape[-1]).T
            M = m.clone().unsqueeze(dim=-1).repeat(1, n.shape[-1])
        else:
            ValueError("n and m must be 1D or 2D")
        return N - M

    @torch.no_grad()
    def update(self, free_opt_Vout: List[torch.Tensor], nudge_opt_Vout: List[torch.Tensor], x):
        """Update weights from optimized Node Voltages (free_opt_Vout & nudge_opt_Vout)"""
        self.W.requires_grad_(True)
        self.W.zero_grad()
        self.fdV.clear()
        self.ndV.clear()
        for idx, W in enumerate(self.W):
            free_opt_vin = self.activation(free_opt_Vout[idx - 1]) if idx != 0 else x
            fdV = self.deltaV(free_opt_vin, free_opt_Vout[idx])
            self.fdV.append(fdV.mean(dim=0))
            nudge_opt_vin = self.activation(nudge_opt_Vout[idx - 1]) if idx != 0 else x
            ndV = self.deltaV(nudge_opt_vin, nudge_opt_Vout[idx])
            self.ndV.append(ndV.mean(dim=0))
            W.weight.grad = (1 / self.beta) * (ndV.pow(2).mean(dim=0) - fdV.pow(2).mean(dim=0))

            # TODO: bias?


class AnalogEP2(nn.Module):
    """Direct implementation of analog eqprop.

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        batch_size: int,
        solver: Callable[[nn.Module], AnalogEqPropSolver],
        cfg: list[int] = [784 * 2, 128, 10 * 2],
        beta: float = 0.1,
        bias: bool = False,
        positive_w: bool = True,
        min_w: float = 1e-6,
        max_w: float | None = None,
        max_w_gain: float = 0.28,
        scale_input: int = 2,
        scale_output: int = 2,
    ) -> None:
        super().__init__()
        self.beta = beta
        layers = []
        for idx in range(len(cfg) - 1):
            layers.append(nn.Linear(cfg[idx], cfg[idx + 1], bias=bias))
        self.model = nn.Sequential(*layers)

        # init weights
        if positive_w:
            self.model.apply(
                eqprop_util.init_params(
                    min_w=min_w,
                    max_w=max_w,
                    max_w_gain=max_w_gain,
                )
            )

        # Add free/nudge nodes per layer as buffers
        self.init_nodes(batch_size)
        self.model.register_buffer("ypred", torch.empty(batch_size, cfg[-1]))

        # instiantiate solver
        self.solver = solver(self.model)

        eqprop_util.interleave.set_num_output(scale_input)
        eqprop_util.interleave.set_num_output(scale_output)

        FutureWarning("AnalogEP2 will be replaced by eqprop.nn.EqPropLinear")

    @eqprop_util.interleave(type="both")
    @torch.no_grad()
    def forward(self, x):
        """Forward propagation.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        # assert self.training is False
        self.reset_nodes()
        reversed_nodes, _ = self.solver(x)
        self.set_nodes(reversed_nodes, positive_phase=True)
        logits = self.model[-1].get_buffer("positive_node")
        self.model.ypred = logits.detach().clone().requires_grad_(True)
        return self.model.ypred

    @eqprop_util.interleave(type="in")
    @torch.no_grad()
    def eqprop(self, x: torch.Tensor):
        """Nudge phase & grad calculation."""
        assert self.training
        reversed_nodes, _ = self.solver(x, nudge_phase=True)
        self.set_nodes(reversed_nodes, positive_phase=False)
        self.prev_positive = self.prev_negative = x
        self.model.apply(self._update)

    def _update(self, submodule: nn.Module):
        """Set gradients of parameters manually.

        dL/dw = (nudge_dV^2 - free_dV^2)/beta
        = [prev_negative^2 - n_node^2
        + prev_positive^2 - p_node^2
        - 2(prev_negative.T@n_node - prev_positive@p_node)]/beta

        Args:
            submodule (nn.Module): submodule of self.model
        """
        if hasattr(submodule, "weight"):
            if submodule.weight.grad is None:
                submodule.weight.grad = torch.zeros_like(submodule.weight)
            p_node = submodule.get_buffer("positive_node")
            n_node = submodule.get_buffer("negative_node")
            res = 2 * (
                torch.bmm(p_node.unsqueeze(2), self.prev_positive.unsqueeze(1))
                .reshape((p_node.size(0), *submodule.weight.shape))
                .mean(dim=0)
                - torch.bmm(n_node.unsqueeze(2), self.prev_negative.unsqueeze(1))
                .reshape((p_node.size(0), *submodule.weight.shape))
                .mean(dim=0)
            )
            # broadcast to 2D
            res += self.prev_negative.pow(2).mean(dim=0) - self.prev_positive.pow(2).mean(dim=0)
            res += (n_node.pow(2).mean(dim=0) - p_node.pow(2).mean(dim=0)).unsqueeze(1)
            submodule.weight.grad += res / self.solver.beta
            if submodule.bias is not None:
                if submodule.bias.grad is None:
                    submodule.bias.grad = torch.zeros_like(submodule.bias)
                submodule.bias.grad += (
                    (
                        (n_node - p_node)
                        * (
                            n_node + p_node - 2 * torch.ones_like(p_node)
                        )  # (n-1)^2-(f-1)^2=2(n-f)(n+f-2)
                    ).mean(dim=0)
                    * 2
                    / self.solver.beta
                )

            self.prev_positive = p_node
            self.prev_negative = n_node

    def init_nodes(self, batch_size) -> None:
        """Initialize free/nudge nodes."""

        def _init_nodes(submodule: nn.Module):
            if hasattr(submodule, "weight"):
                assert submodule._get_name() in ["Linear"], "Only Linear layer is supported"
                output_size = submodule.weight.shape[0]
                positive_node = torch.zeros((batch_size, output_size))
                negative_node = torch.zeros((batch_size, output_size))
                submodule.register_buffer("positive_node", positive_node)
                submodule.register_buffer("negative_node", negative_node)

        self.model.apply(_init_nodes)

    def reset_nodes(self):
        for buf in self.model.buffers():
            buf = torch.zeros_like(buf)

    def set_nodes(self, reversed_nodes: list, positive_phase: bool) -> None:
        """Set free/nudge nodes to each layer.

        Args:
            reversed_nodes (list): reversed list of free/nudge nodes from last layer to first layer
            positive_phase (bool): True if positive phase, False otherwise
        """

        def _set_nodes_layer(submodule: nn.Module):
            nonlocal reversed_nodes
            if hasattr(submodule, "positive_node"):
                if positive_phase:
                    submodule.positive_node = reversed_nodes.pop()
                else:
                    submodule.negative_node = reversed_nodes.pop()

        self.model.apply(_set_nodes_layer)
        del reversed_nodes


class AnalogEPSym(AnalogEP2):
    """Symmetric version of AnalogEP2.

    Use 3rd nudge phase to compute gradients.
    """

    @eqprop_util.interleave(type="both")
    @torch.no_grad()
    def forward(self, x):
        """Forward propagation."""
        # assert self.training is False
        self.reset_nodes()
        reversed_nodes, _ = self.solver(x)
        logits = reversed_nodes[0]
        self.model.ypred = logits.clone().detach().requires_grad_(True)
        return self.model.ypred

    @eqprop_util.interleave(type="in")
    @torch.no_grad()
    def eqprop(self, x: torch.Tensor):
        """Nudge phases & grad calculation."""
        reversed_nodes, _ = self.solver(x, nudge_phase=True)
        self.set_nodes(reversed_nodes, positive_phase=True)
        self.solver.beta = "flip"
        reversed_nodes, _ = self.solver(x, nudge_phase=True)
        self.set_nodes(reversed_nodes, positive_phase=False)
        self.prev_positive = self.prev_negative = x
        self.model.apply(self._update)
        self.solver.beta = "flip"


class DummyAnalogEP2(AnalogEP2):
    """Dummy AnalogEP2 for testing purposes."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model.insert(1, nn.ReLU())

    @eqprop_util.interleave(type="both")
    def forward(self, x):
        return self.model(x)

    def eqprop(self, x: torch.Tensor):
        pass
