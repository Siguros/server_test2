from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eqprop import eqprop_util
from src.utils import get_pylogger

log = get_pylogger(__name__)


class AbstractStrategy(ABC):
    """Abstract class for different strategies to solve for the equilibrium point of the
    network."""

    def __init__(
        self,
        activation: Callable | str,
        model: nn.Module = None,
        max_iter: int = 30,
        atol: float = 1e-6,
        **kwargs,
    ):
        self.activation = activation
        self.model = model
        self.max_iter = max_iter
        self.atol = atol

    @abstractmethod
    def solve(self, x, i_ext, **kwargs) -> list[torch.Tensor]:
        """Solve for the equilibrium point of the network.

        Args:
            x (_type_): input of the network.
            i_ext (_type_): external current.

        Returns:
            list[torch.Tensor]: list of layer node potentials.
        """
        ...

    def get_params(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Returns the weights and biases of the model as a tuple of two lists.

        Each list contains torch.Tensor objects representing the weights and biases of a layer in
        the model.
        """
        if hasattr(self, "W"):
            return (self.W, self.B)
        else:
            self.W, self.B, self.dims = [], [], []
            for name, param in self.model.named_parameters():
                if name.endswith("weight"):
                    self.W.append(param)
                    self.dims.append(param.shape[0])
                elif name.endswith("bias"):
                    self.B.append(param)
            return (self.W, self.B)

    def set_model(self, model: nn.Module) -> None:
        """Set the model."""
        self.model = model


class SPICEStrategy(AbstractStrategy):
    """Calculate Node potentials with SPICE."""

    def __new__(cls):
        # check if ngspice is installed
        cls._check_spice()
        return super().__new__(cls)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _check_spice(cls):
        raise NotImplementedError()


class TorchStrategy(AbstractStrategy):
    """Calculate Node potentials with PyTorch."""

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        """Initialize the solver."""
        self.OTS = self.activation
        self.dims = None
        self.amp_factor = kwargs.get("amp_factor", 1.0)
        self.attrchecked: bool = False
        # self.sparse = (
        #     True if sum([dim**2 for dim in self.dims]) / sum(self.dims) ** 2 < 0.1 else False
        # )

    def check_and_set_attrs(self, kwargs: dict):
        """Check if all attributes are set and set them if not."""
        if self.attrchecked:
            return
        else:
            self.attrchecked = True
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                log.warning(f"key {key} not found in {self.__class__.__name__}")
        for attr in ["OTS", "dims"]:
            if getattr(self, attr) is None:
                raise ValueError(f"{attr} must be set before calling")


class XYCEStrategy(SPICEStrategy):
    """Calculate Node potentials with Xyce."""

    # TODO: check if xyce is installed
    @classmethod
    def _check_spice(cls):
        ...
        # if _XYCE_AVAILABLE:
        #     pass
        # else:
        #     raise ImportError("Xyce is not installed.")

    # TODO: set up xyce, generate netlist
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    # TODO: return nodes in equilibrium
    def solve(self, **kwargs) -> list[torch.Tensor]:
        nodes = 0
        return nodes


class NewtonStrategy(TorchStrategy):
    r"""Solve J\Delta{X}=-f with Newton's method."""

    def __init__(self, clip_threshold, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_threshold = clip_threshold
        self._free_solution = None

    @torch.no_grad()
    def solve(self, x: torch.Tensor, i_ext, **kwargs) -> list[torch.Tensor]:
        """Solve for the equilibrium point of the network.

        Args:
            x (torch.Tensor): input of the network.
            i_ext (torch.Tensor): external current.
        KwArgs:
            params (tuple[list[torch.Tensor], list[torch.Tensor]]): weights and biases of the model.
            OTS (eqprop_util.P3OTS): nonlinearity.
            dims (list): dimensions of the model.
            max_iter (int): maximum number of iterations.
            atol (float): absolute tolerance.
            amp_factor (float): inter-layer potential amplifying factor.
        """
        (W, B) = kwargs.get("params", self.get_params())
        self.check_and_set_attrs(kwargs)
        if i_ext is None:
            self._free_solution = None

        if type(self.OTS) == eqprop_util.SymOTS:
            vout = self._densecholsol2(x, W, B, i_ext)
        else:
            vout = self._densecholsol1(x, W, B, i_ext)
        if i_ext is None:
            self._free_solution = vout.detach().clone()
        nodes = list(vout.split(self.dims, dim=1))
        return nodes

    @torch.no_grad()
    def _densecholsol(
        self,
        x: torch.Tensor,
        W: list,
        B: list | None = None,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J\Delta{X}=-f with dense cholesky/LU decomposition.

        Args:
            x (torch.Tensor): Input.
            W (list): List of weight matrices.
            dims (list): List of dimensions.
            B (list, optional): List of bias vectors. Defaults to None.
            i_ext ([type], optional): External current. Defaults to None.
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_util.self.OTS().
            self.max_iter (int, optional): Maximum number of iterations. Defaults to 30.
            self.atol (float, optional): Absolute tolerance. Defaults to 1e-6.
            self.amp_factor (float, optional): Layerwise voltage&current amplitude factor. Defaults to 1.0.
        """
        dims = self.dims
        batchsize = x.size(0)
        size = sum(dims)
        # construct the laplacian
        paddedG = [torch.zeros(dims[0], size).type_as(x)]
        [
            paddedG.append(F.pad(-g, (sum(dims[:i]), sum(dims[1 + i :]))))
            for i, g in enumerate(W[1:])
        ]
        Ll = torch.cat(paddedG, dim=-2)
        L = Ll + Ll.mT * self.amp_factor
        # construct the RHS
        B = (
            torch.zeros((x.size(0), size)).type_as(x)
            if not B
            else torch.cat(B, dim=-1).unsqueeze(0).repeat(batchsize, 1)
        )
        B[:, : dims[0]] += x @ W[0].T
        if i_ext is not None:
            B[:, -dims[-1] :] += i_ext
        B *= self.amp_factor
        # construct the diagonal
        D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(W[0].sum(-1), (0, size - dims[0]))
        L += D0.diag()
        # initial solution
        if self._free_solution is not None and i_ext is not None:
            v = self._free_solution.detach().clone()
        else:
            lo, info = torch.linalg.cholesky_ex(L)
            v = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
        residual = torch.ones(1).type_as(x)
        L = L.expand(batchsize, *L.shape)
        idx = 1
        while (residual.abs().max() > self.atol) and (idx < self.max_iter):
            # nonlinearity comes here
            # A = self.OTS.a(v[:, : -dims[-1]])
            A = self.OTS.a(v)
            J = L.clone()
            # J[:, : -dims[-1], : -dims[-1]] += torch.stack([a.diag() for a in A])  # expensive?
            # J += torch.stack([a.diag() for a in A])  # expensive?
            J.diagonal(dim1=1, dim2=2)[:] += A
            residual = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
            # residual[:, : -dims[-1], 0] += self.OTS.i(v[:, : -dims[-1]])
            residual[:, :, 0] += self.OTS.i(v)
            # or SPOSV
            if self.amp_factor == 1.0:
                lo, info = torch.linalg.cholesky_ex(J)
                if any(info):  # singular
                    log.debug(f"J is singular, info={info}")
                    lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
                    dv = torch.linalg.lu_solve(lo, piv, -residual).squeeze(-1)
                else:
                    dv = torch.cholesky_solve(-residual, lo).squeeze(-1)
            else:
                lo, piv, info = torch.linalg.lu_factor_ex(J)
                if any(info):
                    lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
                    dv = torch.linalg.lu_solve(lo, piv, -residual).squeeze(-1)
                else:
                    dv = torch.linalg.lu_solve(lo, piv, -residual).squeeze(-1)
            # limit the voltage change
            dv = dv.clamp(min=-self.clip_threshold, max=self.clip_threshold)
            idx += 1
            v += dv

        log.debug(f"condition number of J: {torch.linalg.cond(J[0]):.2f}")
        if idx == self.max_iter:
            log.warning(
                f"stepsolve did not converge in {self.max_iter} iterations, residual={residual.abs().max():.3e}"
            )
        else:
            log.debug(f"stepsolve converged in {idx} iterations")
        return v

    @torch.no_grad()
    def _densecholsol1(
        self,
        x: torch.Tensor,
        W: list,
        B: list | None = None,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve with normalized laplacian.

        Args:
            x (torch.Tensor): Input.
            W (list): List of weight matrices.
            dims (list): List of dimensions.
            B (list, optional): List of bias vectors. Defaults to None.
            i_ext ([type], optional): External current. Defaults to None.
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_util.self.OTS().
            self.max_iter (int, optional): Maximum number of iterations. Defaults to 30.
            self.atol (float, optional): Absolute tolerance. Defaults to 1e-6.
            self.amp_factor (float, optional): Layerwise voltage&current amplitude factor. Defaults to 1.0.
        """
        dims = self.dims
        batchsize = x.size(0)
        size = sum(dims)
        # construct the laplacian
        paddedG = [torch.zeros(dims[0], size).type_as(x)]
        [
            paddedG.append(F.pad(-g, (sum(dims[:i]), sum(dims[1 + i :]))))
            for i, g in enumerate(W[1:])
        ]
        Ll = torch.cat(paddedG, dim=-2)
        mG = Ll + Ll.mT * self.amp_factor
        # construct the RHS
        B = (
            torch.zeros((x.size(0), size)).type_as(x)
            if not B
            else torch.cat(B, dim=-1).unsqueeze(0).repeat(batchsize, 1)
        )
        B[:, : dims[0]] += x @ W[0].T
        if i_ext is not None:
            B[:, -dims[-1] :] += i_ext
        B *= self.amp_factor
        # construct the diagonal
        D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(W[0].sum(-1), (0, size - dims[0]))
        D_inv = 1 / D0
        L = torch.eye(size).type_as(x) + D_inv * mG
        B *= D_inv
        if self._free_solution is not None and i_ext is not None:
            v = self._free_solution.detach().clone()
        else:
            lo, info = torch.linalg.cholesky_ex(L)
            v = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
        residual = torch.ones(1).type_as(x)
        L = L.expand(batchsize, *L.shape)
        idx = 1
        while (residual.abs().max() > self.atol) and (idx < self.max_iter):
            # nonlinearity comes here
            A = self.OTS.a(v)
            J = L.clone()
            J.diagonal(dim1=1, dim2=2)[:] += D_inv * A
            residual = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)

            residual[:, :, 0] += self.OTS.i(v)
            # or SPOSV
            if self.amp_factor == 1.0:
                lo, info = torch.linalg.cholesky_ex(J)
                if any(info):  # singular
                    log.debug(f"J is singular, info={info}")
                    lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
                    dv = torch.linalg.lu_solve(lo, piv, -residual).squeeze(-1)
                else:
                    dv = torch.cholesky_solve(-residual, lo).squeeze(-1)
            else:
                lo, piv, info = torch.linalg.lu_factor_ex(J)
                if any(info):
                    lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
                    dv = torch.linalg.lu_solve(lo, piv, -residual).squeeze(-1)
                else:
                    dv = torch.linalg.lu_solve(lo, piv, -residual).squeeze(-1)
            # limit the voltage change
            dv = dv.clamp(min=-self.clip_threshold, max=self.clip_threshold)
            idx += 1
            v += dv

        log.debug(f"condition number of J: {torch.linalg.cond(J[0]):.2f}")
        if idx == self.max_iter:
            log.warning(
                f"stepsolve did not converge in {self.max_iter} iterations, residual={residual.abs().max():.3e}, vmax={v.abs().max():.3e}"
            )
        else:
            log.debug(f"stepsolve converged in {idx} iterations")
        return v

    @torch.no_grad()
    def _densecholsol2(
        self,
        x: torch.Tensor,
        W: list,
        B: list | None = None,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J\Delta{X}=-f with robust dense cholesky/LU decomposition.

        Args:
            x (torch.Tensor): Input.
            W (list): List of weight matrices.
            dims (list): List of dimensions.
            B (list, optional): List of bias vectors. Defaults to None.
            i_ext ([type], optional): External current. Defaults to None.
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_util.self.OTS().
            self.max_iter (int, optional): Maximum number of iterations. Defaults to 30.
            self.atol (float, optional): Absolute tolerance. Defaults to 1e-6.
            self.amp_factor (float, optional): Layerwise voltage&current amplitude factor. Defaults to 1.0.
        """
        dims = self.dims
        batchsize = x.size(0)
        size = sum(dims)
        # construct the laplacian
        paddedG = [torch.zeros(dims[0], size).type_as(x)]
        [
            paddedG.append(F.pad(-g, (sum(dims[:i]), sum(dims[1 + i :]))))
            for i, g in enumerate(W[1:])
        ]
        Ll = torch.cat(paddedG, dim=-2)
        L = Ll + Ll.mT * self.amp_factor
        # construct the RHS
        B = (
            torch.zeros((x.size(0), size)).type_as(x)
            if not B
            else torch.cat(B, dim=-1).unsqueeze(0).repeat(batchsize, 1)
        )
        B[:, : dims[0]] += x @ W[0].T
        if i_ext is not None:
            B[:, -dims[-1] :] += i_ext
        B *= self.amp_factor
        # construct the diagonal
        D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(W[0].sum(-1), (0, size - dims[0]))
        L += D0.diag()
        # initial solution
        if self._free_solution is not None and i_ext is not None:
            v = self._free_solution.detach().clone()
        else:
            lo, info = torch.linalg.cholesky_ex(L)
            v = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
        residual = torch.ones(1).type_as(x)
        L = L.expand(batchsize, *L.shape)
        idx = 1
        while (residual.abs().max() > self.atol) and (idx < self.max_iter):
            # nonlinearity comes here
            # log.debug(f"absvmax: {v.abs().max():.3e}, absvmin: {v.abs().min():.3e}")
            a_inv = (1 / self.OTS.a(v)).unsqueeze(-1)
            J = a_inv * (L.clone()) + torch.eye(L.size(-1))  # L shape, a_inv shape?
            residual = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
            residual2 = a_inv * residual
            residual2[:, :, 0] += self.OTS.i_div_a(v)
            log.debug(f"residual: {residual.abs().max():.3e}")
            # or SPOSV
            lo, info = torch.linalg.cholesky_ex(J)
            if any(info):  # singular
                log.debug(f"J is singular, info={info}")
                dv = torch.linalg.lstsq(J, -residual2, driver="gels").solution.squeeze()
            else:
                dv = torch.cholesky_solve(-residual2, lo).squeeze(-1)

            # limit the voltage change
            dv = dv.clamp(min=-self.clip_threshold, max=self.clip_threshold)
            idx += 1
            v += dv

        log.debug(f"condition number of J: {torch.linalg.cond(J[0]):.2f}")
        if idx == self.max_iter:
            log.warning(
                f"stepsolve did not converge in {self.max_iter} iterations, dv={dv.abs().max():.3e}, residual={residual.abs().max():.3e}"
            )
        else:
            log.debug(f"stepsolve converged in {idx} iterations")
        return v

    @torch.no_grad()
    def _leastsqsol(
        self,
        x: torch.Tensor,
        W: list,
        B: list | None = None,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J^TJ\Delta{X}=-J^Tf with least square method.

        Args:
            x (torch.Tensor): Input.
            W (list): List of weight matrices.
            dims (list): List of dimensions.
            B (list, optional): List of bias vectors. Defaults to None.
            i_ext ([type], optional): External current. Defaults to None.
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_util.self.OTS().
            self.max_iter (int, optional): Maximum number of iterations. Defaults to 30.
            self.atol (float, optional): Absolute tolerance. Defaults to 1e-6.
            self.amp_factor (float, optional): Layerwise voltage&current amplitude factor. Defaults to 1.0.
        """
        dims = self.dims
        batchsize = x.size(0)
        size = sum(dims)
        # construct the laplacian
        paddedG = [torch.zeros(dims[0], size).type_as(x)]
        [
            paddedG.append(F.pad(-g, (sum(dims[:i]), sum(dims[1 + i :]))))
            for i, g in enumerate(W[1:])
        ]
        Ll = torch.cat(paddedG, dim=-2)
        L = Ll + Ll.mT * self.amp_factor
        # construct the RHS
        B = (
            torch.zeros((x.size(0), size)).type_as(x)
            if not B
            else torch.cat(B, dim=-1).unsqueeze(0).repeat(batchsize, 1)
        )
        B[:, : dims[0]] += x @ W[0].T
        if i_ext is not None:
            B[:, -dims[-1] :] += i_ext
        B *= self.amp_factor
        # construct the diagonal
        D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(W[0].sum(-1), (0, size - dims[0]))
        L += D0.diag()
        # initial solution
        # lo, info = torch.linalg.cholesky_ex(L)
        # v = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
        L = L.expand(batchsize, *L.shape)
        # initial solution
        if self._free_solution is not None and i_ext is not None:
            v = self._free_solution.detach().clone()
        else:
            v = torch.linalg.lstsq(L, B.unsqueeze(-1)).solution.squeeze()
        dv = torch.ones(1).type_as(x)
        # L = L.expand(batchsize, *L.shape)
        idx = 1
        while (dv.abs().max() > self.atol) and (idx < self.max_iter):
            # nonlinearity comes here
            A = self.OTS.a(v[:, : -dims[-1]])
            J = L.clone()
            J[:, : -dims[-1], : -dims[-1]] += torch.stack([a.diag() for a in A])  # expensive?
            f = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
            f[:, : -dims[-1], 0] += self.OTS.i(v[:, : -dims[-1]])
            # or SPOSV
            # lo, info = torch.linalg.cholesky_ex(J)
            # if any(info):  # singular
            #     log.debug(f"J is singular, info={info}")
            #     lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
            #     dv = torch.linalg.lu_solve(lo, piv, -f).squeeze(-1)
            # else:
            #     dv = torch.cholesky_solve(-f, lo).squeeze(-1)
            res = torch.linalg.lstsq(J, -f, driver="gels")
            dv = res.solution.squeeze()
            # limit the voltage change
            dv = dv.clamp(min=-self.clip_threshold, max=self.clip_threshold)
            idx += 1
            v += dv

        log.debug(f"condition number of J: {torch.linalg.cond(J[0]):.2f}")
        if idx == self.max_iter:
            log.warning(
                f"stepsolve did not converge in {self.max_iter} iterations, dv={dv.abs().max():.3e}"
            )
        else:
            log.debug(f"stepsolve converged in {idx} iterations")
        return v

    @torch.no_grad()
    def _precondlusol(
        self,
        x: torch.Tensor,
        W: list,
        dims: list,
        B: list | None = None,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J\Delta{X}=-f with diag-preconditioned LU decomposition."""
        batchsize = x.size(0)
        size = sum(dims)
        # construct the laplacian
        paddedG = [torch.zeros(dims[0], size).type_as(x)]
        for i, g in enumerate(W[1:]):
            paddedG.append(F.pad(-g, (sum(dims[:i]), sum(dims[1 + i :]))))

        Ll = torch.cat(paddedG, dim=-2)
        L = Ll + Ll.mT
        # construct the RHS
        B = (
            torch.zeros((x.size(0), size)).type_as(x)
            if not B
            else torch.cat(B, dim=-1).unsqueeze(0).repeat(batchsize, 1)
        )
        B[:, : dims[0]] += x @ W[0].T
        B[:, -dims[-1] :] += i_ext
        # construct the diagonal
        D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(W[0].sum(-1), (0, size - dims[0]))
        P_inv = 1 / D0
        L += D0.diag()
        # initial solution
        lo, info = torch.linalg.cholesky_ex(L * P_inv)
        y = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
        v = P_inv * y
        dv = torch.ones(1).type_as(x)
        L = L.expand(batchsize, *L.shape)
        idx = 1
        while (dv.abs().max() > self.atol) and (idx < self.max_iter):
            # nonlinearity comes here
            A = self.OTS.a(v[:, : -dims[-1]])
            J = L.clone()
            J[:, : -dims[-1], : -dims[-1]] += torch.stack([a.diag() for a in A])
            f = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
            f[:, : -dims[-1], 0] += self.OTS.i(v[:, : -dims[-1]])
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
        if idx == self.max_iter:
            log.warning(
                f"stepsolve did not converge in {self.max_iter} iterations, dv={dv.abs().max():.3e}"
            )
        else:
            log.debug(f"stepsolve converged in {idx} iterations")
        return v


class LMStrategy(TorchStrategy):
    r"""Solve J\Delta{X}=-f with Levenberg-Marquardt method."""

    def __init__(self, clip_threshold, lambda_factor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_threshold = clip_threshold
        self.lambda_factor = lambda_factor
        self._free_solution = None

    @torch.no_grad()
    def solve(self, x, i_ext, **kwargs) -> list[torch.Tensor]:
        """Solve for the equilibrium point of the network.

        Args:
            x (torch.Tensor): input of the network.
            i_ext (torch.Tensor): external current.
        KwArgs:
            params (tuple[list[torch.Tensor], list[torch.Tensor]]): weights and biases of the model.
            OTS (eqprop_util.P3OTS): nonlinearity.
            dims (list): dimensions of the model.
            max_iter (int): maximum number of iterations.
            atol (float): absolute tolerance.
            amp_factor (float): inter-layer potential amplifying factor.
        """
        self.check_and_set_attrs(kwargs)
        (W, B) = kwargs.get("params", self.get_params())
        if i_ext is None:
            self._free_solution = None
        vout = self._LMdensecholsol(x, W, B, i_ext)
        if i_ext is None:
            self._free_solution = vout.detach().clone()
        nodes = list(vout.split(self.dims, dim=1))
        return nodes

    def _LMdensecholsol(
        self,
        x: torch.Tensor,
        W: list,
        B: list | None = None,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J\Delta{X}=-f with dense cholesky decomposition.

        Args:
            x (torch.Tensor): Input.
            W (list): List of weight matrices.
            dims (list): List of dimensions.
            B (list, optional): List of bias vectors. Defaults to None.
            i_ext ([type], optional): External current. Defaults to None.
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_util.self.OTS().
            self.max_iter (int, optional): Maximum number of iterations. Defaults to 30.
            self.atol (float, optional): Absolute tolerance. Defaults to 1e-6.
            self.amp_factor (float, optional): Layerwise voltage&current amplitude factor. Defaults to 1.0.
        """
        dims = self.dims
        batchsize = x.size(0)
        size = sum(dims)
        lambda_ = self.lambda_factor
        # construct the laplacian
        paddedG = [torch.zeros(dims[0], size).type_as(x)]
        [
            paddedG.append(F.pad(-g, (sum(dims[:i]), sum(dims[1 + i :]))))
            for i, g in enumerate(W[1:])
        ]
        Ll = torch.cat(paddedG, dim=-2)
        L = Ll + Ll.mT * self.amp_factor
        # construct the RHS
        B = (
            torch.zeros((x.size(0), size)).type_as(x)
            if not B
            else torch.cat(B, dim=-1).unsqueeze(0).repeat(batchsize, 1)
        )
        B[:, : dims[0]] += x @ W[0].T
        if i_ext is not None:
            B[:, -dims[-1] :] += i_ext
        B *= self.amp_factor
        # construct the diagonal
        D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(W[0].sum(-1), (0, size - dims[0]))
        L += D0.diag()
        # initial solution
        # lo, info = torch.linalg.cholesky_ex(L)
        # v = torch.cholesky_solve(B.unsqueeze(-1), lo).squeeze(-1)
        L = L.expand(batchsize, *L.shape)
        # initial solution
        if self._free_solution is not None and i_ext is not None:
            v = self._free_solution.detach().clone()
        else:
            v = torch.linalg.lstsq(L, B.unsqueeze(-1)).solution.squeeze()
        dv = torch.ones(1).type_as(x)
        # L = L.expand(batchsize, *L.shape)
        idx = 1
        residuals = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
        residuals[:, :, 0] += self.OTS.i(v)
        while idx < self.max_iter:
            # nonlinearity comes here
            A = self.OTS.a(v)
            jacobian = L.clone()
            jacobian += torch.stack([a.diag() for a in A])  # expensive?

            if torch.all(torch.norm(residuals, dim=(1, 2)) < self.atol):
                break

            # Update rule
            JTJ = torch.bmm(jacobian.mT, jacobian)
            while True:
                try:
                    # Try to update parameters
                    dv = -torch.linalg.solve(
                        torch.bmm(jacobian.mT, residuals),
                        JTJ + lambda_ * torch.eye(v.size(-1)),
                    )
                    break
                except RuntimeError:
                    # In case of singular matrix, increase damping and try again
                    log.debug(f"Jacobian is singular, lambda={lambda_}")
                    lambda_ *= 10
                    if lambda_ > 1e2:
                        raise RuntimeError("Jacobian is singular")
            # or SPOSV
            # lo, info = torch.linalg.cholesky_ex(J)
            # if any(info):  # singular
            #     log.debug(residuals"J is singular, info={info}")
            #     lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
            #     dv = torch.linalg.lu_solve(lo, piv, -residuals).squeeze(-1)
            # else:
            #     dv = torch.cholesky_solve(-residuals, lo).squeeze(-1)

            v += dv
            new_residuals = torch.bmm(L, v.unsqueeze(-1)) - B.clone().unsqueeze(-1)
            new_residuals[:, : -dims[-1], 0] += self.OTS.i(v[:, : -dims[-1]])
            if torch.all(
                torch.norm(new_residuals, dim=(1, 2)) < torch.norm(residuals, dim=(1, 2))
            ):
                lambda_ /= 10

            idx += 1
            residuals = new_residuals

        log.debug(f"condition number of J: {torch.linalg.cond(jacobian[0]):.2f}")
        if idx == self.max_iter:
            log.warning(
                f"stepsolve did not converge in {self.max_iter} iterations, dv={dv.abs().max():.3e}"
            )
        else:
            log.debug(f"stepsolve converged in {idx} iterations")
        return v


def _inv_L(W: torch.Tensor, D0: torch.Tensor, dims: list) -> torch.Tensor:
    hidden_len = dims[0]
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
def _block_tri_cholesky(W: list[torch.Tensor]):
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
