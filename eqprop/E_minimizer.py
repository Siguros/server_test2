# from __future__ import annotations
from typing import Generator, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.models.components.eqprop_backbone import AnalogEP2
from src.eqprop import eqprop_util
from src.utils import get_pylogger

log = get_pylogger(__name__)


# functions below are used as instance methods
def newton_solver(
    self,
    x,
    y=None,
    Nodes: List[torch.Tensor] = None,
    beta=0.0,
    iters=None,
    training=True,
) -> List[torch.Tensor]:
    """Find node voltages by linearize & iteratively solve network to satisfy KCL. Used as a method
    of AnalogEP class.

    Args:
        x (_type_): _description_
        y (_type_, optional): _description_. Defaults to None.
        Nodes (List[torch.Tensor], optional): _description_. Defaults to None.
        beta (float, optional): _description_. Defaults to 0.0.
        iters (_type_, optional): _description_. Defaults to None.
        training (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    dims = self.dims
    self.W.requires_grad_(False)
    if beta != 0:
        self.ypred.requires_grad_(True)
        loss: torch.Tensor = self.loss(self.ypred, y)
        loss.sum().backward()
        i_ext = -beta * self.ypred.grad
    else:
        i_ext = 0
    vout = _stepsolve(x, self.W, dims, i_ext=i_ext)
    Nodes = list(vout.split(dims[1:], dim=1))
    self.ypred = Nodes[-1].clone().detach() if beta == 0 else None
    return Nodes


# TODO: implement this
class newtonSolver:
    def __init__(
        self, OTS: eqprop_util.OTS = eqprop_util.OTS(), max_iter: int = 30, atol: float = 1e-6
    ) -> None:
        self.OTS = OTS
        self.max_iter = max_iter
        self.atol = atol

    def set_params_from_net(self, net: "AnalogEP2") -> None:  # noqa F821
        self.dims = net.dims
        self.sparse = (
            True if sum([dim**2 for dim in self.dims]) / sum(self.dims) ** 2 < 0.1 else False
        )
        assert hasattr(net.model, "ypred"), ValueError("model must have a ypred attribute")
        self.model = net.model
        self.beta = net.beta

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: Union[None, torch.Tensor] = None) -> None:
        assert hasattr(self, "model"), ValueError("model must be set before calling")
        free_phase = True if y is None else False
        if free_phase:
            self.W, self.B = [], []
            self.model.apply(self.get_params)
            i_ext = 0
        else:
            assert self.W is not None and self.B is not None, ValueError(
                "W and B must be exist in free phase"
            )
            i_ext = -self.beta * self.model.ypred.grad
            del self.model.ypred.grad
        vout = (
            _stepsolve2(
                x,
                self.W,
                self.dims,
                self.B,
                i_ext=i_ext,
                OTS=self.OTS,
                max_iter=self.max_iter,
                atol=self.atol,
            )
            if not self.sparse
            else _sparsesolve(x, self.W, self.B, i_ext)
        )
        Nodes = list(vout.split(self.dims[1:], dim=1))
        Nodes.reverse()
        return Nodes

    def get_params(self, submodule: nn.Module):
        if hasattr(submodule, "weight"):
            self.W.append(submodule.get_parameter("weight"))
            self.B.append(submodule.get_parameter("bias")) if submodule.bias is not None else ...


@torch.no_grad()
def _stepsolve(
    x: torch.Tensor, W: torch.nn.ModuleList, dims, i_ext=0, atol=1e-6, it=30
) -> torch.Tensor:
    r"""Solve J\Delta{X}=-f iteraitvely."""
    if not hasattr(_stepsolve, "rectifier"):
        _stepsolve.rectifier = eqprop_util.OTS(Is=1e-6, Vr=0.9, Vl=0.1)
    batchsize = x.size(0)
    size = sum(dims[1:])
    paddedG = [torch.zeros(dims[1], size).type_as(x)]
    [
        paddedG.append(F.pad(-g.weight, (sum(dims[1 : i + 1]), sum(dims[2 + i :]))))
        for i, g in enumerate(W[1:])
    ]
    Ll = torch.cat(paddedG, dim=-2)
    L = Ll + Ll.mT
    B = torch.zeros((x.size(0), size)).type_as(x)
    B[:, : dims[1]] = x @ W[0].weight.T
    B[:, -dims[-1] :] = i_ext
    D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(W[0].weight.sum(-1), (0, size - dims[1]))
    L += D0.diag()
    lo, info = torch.linalg.cholesky_ex(L)
    v = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
    dv = torch.ones(1).type_as(x)
    L = L.expand(batchsize, *L.shape)  #
    idx = 1
    while (dv.abs().max() > atol) and (idx < it):
        # nonlinearity comes here
        A = _stepsolve.rectifier.a(
            v[:, : -dims[-1]],
            Is=_stepsolve.rectifier.Is,
            Vr=_stepsolve.rectifier.Vr,
            Vl=_stepsolve.rectifier.Vl,
        )
        J = L.clone()
        J[:, : -dims[-1], : -dims[-1]] += torch.stack([a.diag() for a in A])
        f = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
        f[:, : -dims[-1], 0] += _stepsolve.rectifier.i(
            v[:, : -dims[-1]],
            Is=_stepsolve.rectifier.Is,
            Vr=_stepsolve.rectifier.Vr,
            Vl=_stepsolve.rectifier.Vl,
        )
        # or SPOSV()
        lo, info = torch.linalg.cholesky_ex(J)
        dv = torch.cholesky_solve(-f, lo).squeeze(-1)
        thrs = 1e-1  # / idx
        dv = dv.clamp(min=-thrs, max=thrs)  # voltage limit
        idx += 1
        v += dv
    log.debug(f"stepsolve converged in {idx} iterations")
    return v


@torch.no_grad()
def _stepsolve2(
    x: torch.Tensor,
    W: list,
    dims: list,
    B: list | None = None,
    i_ext=None,
    OTS=eqprop_util.OTS(),
    max_iter=30,
    atol=1e-6,
) -> torch.Tensor:
    r"""Solve J\Delta{X}=-f iteraitvely."""
    batchsize = x.size(0)
    size = sum(dims[1:])
    # construct the laplacian
    paddedG = [torch.zeros(dims[1], size).type_as(x)]
    [
        paddedG.append(F.pad(-g, (sum(dims[1 : i + 1]), sum(dims[2 + i :]))))
        for i, g in enumerate(W[1:])
    ]
    Ll = torch.cat(paddedG, dim=-2)
    L = Ll + Ll.mT
    # construct the RHS
    B = (
        torch.zeros((x.size(0), size)).type_as(x)
        if not B
        else torch.cat(B, dim=-1).unsqueeze(0).repeat(batchsize, 1)
    )
    B[:, : dims[1]] += x @ W[0].T
    B[:, -dims[-1] :] += i_ext
    # construct the diagonal
    D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(W[0].sum(-1), (0, size - dims[1]))
    L += D0.diag()
    # initial solution
    lo, info = torch.linalg.cholesky_ex(L)
    v = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
    dv = torch.ones(1).type_as(x)
    L = L.expand(batchsize, *L.shape)
    idx = 1
    while (dv.abs().max() > atol) and (idx < max_iter):
        # nonlinearity comes here
        A = OTS.a(v[:, : -dims[-1]])
        J = L.clone()
        J[:, : -dims[-1], : -dims[-1]] += torch.stack([a.diag() for a in A])
        f = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
        f[:, : -dims[-1], 0] += OTS.i(v[:, : -dims[-1]])
        # or SPOSV
        lo, info = torch.linalg.cholesky_ex(J)
        if any(info):  # singular
            lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
            dv = torch.linalg.lu_solve(lo, piv, -f).squeeze(-1)
        else:
            dv = torch.cholesky_solve(-f, lo).squeeze(-1)
        thrs = 1e-1  # / idx
        dv = dv.clamp(min=-thrs, max=thrs)  # voltage limit
        idx += 1
        v += dv
    return v


def _sparsesolve(x, W, dims, B, i_ext):
    """Use torch.block_diag to construct the sparse matrix.

    Args:
        x (_type_): _description_
        W (_type_): _description_
        B (_type_): _description_
        i_ext (_type_): _description_
    """
    raise NotImplementedError


# TODO: custom CUDA kernel
def block_tri_cholesky(W: List[torch.Tensor]):
    """Blockwise cholesky decomposition for a size varying block tridiagonal matrix. see spftrf()
    in LAPACK.

    Args:
        W (List[torch.Tensor]): List of lower triangular blocks.

    Returns:
        L (List[torch.Tensor]): List of lower triangular blocks.
        C (List[torch.Tensor]): List of diagonal blocks. as column vectors.
    """

    n = len(W)
    C = [torch.zeros_like(W[i]) for i in range(n)]
    L = [None] * (n + 1)
    W.append(0)
    L[0] = torch.sqrt(W[0].sum(dim=-1))
    for i in range(n):
        C[i] = W[i] / L[i]  # C[i] = W[i] @ D_prev^-T, trsm()
        D = W[i].sum(dim=-2) + W[i + 1].sum(dim=-1) - torch.bmm(C[i], C[i].mT)
        L[i + 1] = torch.sqrt(D)

    return L, C


def block_tri_cholesky_solve(L, C, B):
    """Blockwise cholesky solve for a size varying block tridiagonal matrix.

    Args:
        L (List[torch.Tensor]): List of lower triangular blocks.
        C (List[torch.Tensor]): List of diagonal blocks.
        B (torch.Tensor): RHS.

    Returns:
        X (torch.Tensor): Solution.
    """

    n = len(L)
    X = torch.zeros_like(B)
    for i in range(n):
        X[:, i * C[i].size(-1) : (i + 1) * C[i].size(-1)] = torch.cholesky_solve(
            B[:, i * C[i].size(-1) : (i + 1) * C[i].size(-1)],
            L[i + 1] + torch.bmm(C[i].transpose(-1, -2), C[i]),
        )

    return X
