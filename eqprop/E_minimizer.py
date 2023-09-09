# from __future__ import annotations
from typing import Generator, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.models.components.eqprop_backbone import AnalogEP2
from src.eqprop import eqprop_util
from src.utils import get_pylogger

log = get_pylogger(__name__)


class NewtonSolver:
    r"""Solve J\Delta{X}=-f with Newton's method."""

    def __init__(
        self,
        OTS: eqprop_util.OTS = eqprop_util.OTS(),
        max_iter: int = 30,
        atol: float = 1e-6,
        amp_factor: float = 1.0,
    ) -> None:
        self.OTS = OTS
        self.max_iter = max_iter
        self.atol = atol
        self.amp_factor = amp_factor

    def set_params_from_net(self, net: "AnalogEP2") -> None:  # noqa F821
        self.dims = net.dims
        self.sparse = (
            True if sum([dim**2 for dim in self.dims]) / sum(self.dims) ** 2 < 0.1 else False
        )
        assert hasattr(net.model, "ypred"), ValueError("model must have a ypred attribute")
        self.model: nn.Module = net.model
        self.beta = net.beta

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, nudge_phase: bool = False) -> None:
        assert hasattr(self, "model"), ValueError("model must be set before calling")
        W, B = [], []
        self.model.apply(lambda submodule: self.get_params(submodule, W, B))
        i_ext = 0
        if nudge_phase:
            i_ext = -self.beta * self.model.ypred.grad
            self.model.zero_grad()
        vout = (
            _densecholsol(
                x,
                W,
                self.dims,
                B,
                i_ext=i_ext,
                OTS=self.OTS,
                max_iter=self.max_iter,
                atol=self.atol,
                amp_factor=self.amp_factor,
            )
            if not self.sparse
            else _sparsecholsol(x, W, B, i_ext)
        )
        Nodes = list(vout.split(self.dims[1:], dim=1))
        Nodes.reverse()
        return Nodes

    def get_params(
        self, submodule: nn.Module, W: list[torch.Tensor], B: list[torch.Tensor]
    ) -> None:
        if hasattr(submodule, "weight"):
            W.append(submodule.get_parameter("weight"))
            if submodule.bias is not None:
                B.append(submodule.get_parameter("bias"))


@torch.no_grad()
def _densecholsol(
    x: torch.Tensor,
    W: list,
    dims: list,
    B: list | None = None,
    i_ext=None,
    OTS=eqprop_util.OTS(),
    max_iter=30,
    atol=1e-6,
    amp_factor=1.0,
) -> torch.Tensor:
    r"""Solve J\Delta{X}=-f with dense cholesky decomposition.

    Args:
        x (torch.Tensor): Input.
        W (list): List of weight matrices.
        dims (list): List of dimensions.
        B (list, optional): List of bias vectors. Defaults to None.
        i_ext ([type], optional): External current. Defaults to None.
        OTS ([type], optional): Nonlinearity. Defaults to eqprop_util.OTS().
        max_iter (int, optional): Maximum number of iterations. Defaults to 30.
        atol (float, optional): Absolute tolerance. Defaults to 1e-6.
        amp_factor (float, optional): Layerwise voltage&current amplitude factor. Defaults to 1.0.
    """
    batchsize = x.size(0)
    size = sum(dims[1:])
    # construct the laplacian
    paddedG = [torch.zeros(dims[1], size).type_as(x)]
    [
        paddedG.append(F.pad(-g, (sum(dims[1 : i + 1]), sum(dims[2 + i :]))))
        for i, g in enumerate(W[1:])
    ]
    Ll = torch.cat(paddedG, dim=-2)
    L = Ll + Ll.mT * amp_factor
    # construct the RHS
    B = (
        torch.zeros((x.size(0), size)).type_as(x)
        if not B
        else torch.cat(B, dim=-1).unsqueeze(0).repeat(batchsize, 1)
    )
    B[:, : dims[1]] += x @ W[0].T
    B[:, -dims[-1] :] += i_ext
    B *= amp_factor
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

    log.debug(f"condition number of J: {torch.linalg.cond(J[0]):.2f}")
    if idx == max_iter:
        log.warning(
            f"stepsolve did not converge in {max_iter} iterations, dv={dv.abs().max():.3e}"
        )
    else:
        log.debug(f"stepsolve converged in {idx} iterations")
    return v


@torch.no_grad()
def _precondlusol(
    x: torch.Tensor,
    W: list,
    dims: list,
    B: list | None = None,
    i_ext=None,
    OTS=eqprop_util.OTS(),
    max_iter=30,
    atol=1e-6,
) -> torch.Tensor:
    r"""Solve J\Delta{X}=-f with diag-preconditioned LU decomposition."""
    batchsize = x.size(0)
    size = sum(dims[1:])
    # construct the laplacian
    paddedG = [torch.zeros(dims[1], size).type_as(x)]
    for i, g in enumerate(W[1:]):
        paddedG.append(F.pad(-g, (sum(dims[1 : i + 1]), sum(dims[2 + i :]))))

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
    P_inv = 1 / D0
    L += D0.diag()
    # initial solution
    lo, info = torch.linalg.cholesky_ex(L * P_inv)
    y = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
    v = P_inv * y
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
        Precond_J = J * P_inv
        lo, info = torch.linalg.cholesky_ex(Precond_J)
        if any(info):  # singular
            log.debug(f"J is singular, info={info}")
            lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
            dv = torch.linalg.lu_solve(lo, piv, -f).squeeze(-1)
        else:
            dy = torch.cholesky_solve(-f, lo).squeeze(-1)
            dv = P_inv * dy
        thrs = 1e-1  # / idx
        dv = dv.clamp(min=-thrs, max=thrs)  # voltage limit
        idx += 1
        v += dv

    log.debug(f"condition number of J: {torch.linalg.cond(Precond_J[0]):.2f}")
    if idx == max_iter:
        log.warning(
            f"stepsolve did not converge in {max_iter} iterations, dv={dv.abs().max():.3e}"
        )
    else:
        log.debug(f"stepsolve converged in {idx} iterations")
    return v


def _inv_L(W: torch.Tensor, D0: torch.Tensor, dims: list) -> torch.Tensor:
    hidden_len = dims[1]
    D1 = D0[hidden_len:]
    D2 = D0[:hidden_len]
    schur_L = D1.diag() - W.T @ D2.diag() @ W
    inv_schur_L = schur_L.inverse()
    A = inv_schur_L @ W.T @ D2.diag()
    B = -inv_schur_L @ W.T @ D2.diag() @ W
    AB = torch.cat((A, B), dim=-1)
    CD = torch.cat((B.mT, inv_schur_L), dim=-1)
    inv_L = torch.cat((AB, CD), dim=-2)
    return inv_L


def _sparsecholsol(x, W, dims, B, i_ext):
    """Use torch.block_diag to construct the sparse matrix.

    Args:
        x (_type_): _description_
        W (_type_): _description_
        B (_type_): _description_
        i_ext (_type_): _description_
    """
    raise NotImplementedError


# TODO: custom CUDA kernel
def _block_tri_cholesky(W: List[torch.Tensor]):
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


def _block_tri_cholesky_solve(L, C, B):
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
