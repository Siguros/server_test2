# from __future__ import annotations
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.models.components.eqprop_backbone import AnalogEP2
from src.utils.eqprop_util import (
    rectifier_a,
    rectifier_i,
    rectifier_p3_a,
    rectifier_p3_i,
)

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


def newton_solve2(self, x: torch.Tensor, y: Union[None, torch.Tensor] = None) -> None:
    W, B = [], []
    dims = getattr(self, "dims")
    self.model.requires_grad_(False)

    if not hasattr(self, "sparse"):
        self.sparse = True if sum([dim**2 for dim in dims]) / sum(dims) ** 2 < 0.1 else False

    def get_params(submodule: nn.Module):
        nonlocal W, B
        if hasattr(submodule, "weight"):
            W.append(submodule.get_parameter("weight"))
            B.append(submodule.get_parameter("bias")) if self.hparams["bias"] else ...

    self.model.apply(get_params)
    free_phase = True if y is None else False
    if free_phase:
        i_ext = 0
    else:
        i_ext = -self.beta * self.model.ypred.grad
        del self.model.ypred.grad
    vout = (
        _stepsolve2(x, W, dims, i_ext=i_ext) if not self.sparse else _sparsesolve(x, W, B, i_ext)
    )
    del W, B
    Nodes = list(vout.split(dims[1:], dim=1))
    Nodes.reverse()

    def buffer_setter(submodule: nn.Module):
        if hasattr(submodule, "free_node"):
            if free_phase:
                submodule.free_node = Nodes.pop()
            else:
                submodule.nudge_node = Nodes.pop()

    self.model.apply(buffer_setter)


@torch.no_grad()
def _stepsolve(
    x: torch.Tensor, W: torch.nn.ModuleList, dims, i_ext=0, atol=1e-6, it=30
) -> torch.Tensor:
    r"""Solve J\Delta{X}=-f iteraitvely."""
    if not hasattr(_stepsolve, "Is"):
        _stepsolve.Is = 1e-6
        _stepsolve.Vr = 0.9
        _stepsolve.Vl = 0.1
    b = x.size(0)
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
    L = L.expand(b, *L.shape)  #
    idx = 1
    while (dv.abs().max() > atol) and (idx < it):
        # nonlinearity comes here
        A = rectifier_a(v[:, : -dims[-1]], Is=_stepsolve.Is, Vr=_stepsolve.Vr, Vl=_stepsolve.Vl)
        J = L.clone()
        J[:, : -dims[-1], : -dims[-1]] += torch.stack([a.diag() for a in A])
        f = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
        f[:, : -dims[-1], 0] += rectifier_i(
            v[:, : -dims[-1]], Is=_stepsolve.Is, Vr=_stepsolve.Vr, Vl=_stepsolve.Vl
        )
        # or SPOSV
        lo, info = torch.linalg.cholesky_ex(J)
        dv = torch.cholesky_solve(-f, lo).squeeze(-1)
        thrs = 1e-1  # / idx
        dv = dv.clamp(min=-thrs, max=thrs)  # voltage limit
        idx += 1
        v += dv
    return v


@torch.no_grad()
def _stepsolve2(x: torch.Tensor, W: list, dims: list, B=None, i_ext=None):
    atol = 1e-6
    it = 30
    if not hasattr(_stepsolve, "Is"):
        _stepsolve.Is = 1e-6
        _stepsolve.Vr = 0.9
        _stepsolve.Vl = 0.1
    b = x.size(0)
    size = sum(dims[1:])
    paddedG = [torch.zeros(dims[1], size).type_as(x)]
    [
        paddedG.append(F.pad(-g, (sum(dims[1 : i + 1]), sum(dims[2 + i :]))))
        for i, g in enumerate(W[1:])
    ]
    Ll = torch.cat(paddedG, dim=-2)
    L = Ll + Ll.mT
    B = torch.zeros((x.size(0), size)).type_as(x)
    B[:, : dims[1]] = x @ W[0].T
    B[:, -dims[-1] :] = i_ext
    D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(W[0].sum(-1), (0, size - dims[1]))
    L += D0.diag()
    lo, info = torch.linalg.cholesky_ex(L)
    v = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
    dv = torch.ones(1).type_as(x)
    L = L.expand(b, *L.shape)  #
    idx = 1
    while (dv.abs().max() > atol) and (idx < it):
        # nonlinearity comes here
        A = rectifier_p3_a(v[:, : -dims[-1]], Is=_stepsolve.Is, Vr=_stepsolve.Vr, Vl=_stepsolve.Vl)
        J = L.clone()
        J[:, : -dims[-1], : -dims[-1]] += torch.stack([a.diag() for a in A])
        f = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
        f[:, : -dims[-1], 0] += rectifier_p3_i(
            v[:, : -dims[-1]], Is=_stepsolve.Is, Vr=_stepsolve.Vr, Vl=_stepsolve.Vl
        )
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
    ...


def block_tri_cholesky(W: List[torch.Tensor]):
    """Blockwise cholesky decomposition for a block diagonal matrix.

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
    """Blockwise cholesky solve for a block diagonal matrix.

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
