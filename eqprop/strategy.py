import os
import shutil
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import fsolve

from src.eqprop import eqprop_util
from src.utils import get_pylogger

if _XYCE_AVAILABLE := shutil.which("Xyce"):
    from src.eqprop.xyce_util.circuits import _createCircuit
    from src.eqprop.xyce_util.MyCircuit import MyCircuit
    from src.eqprop.xyce_util.NetlistWriter import SPICEParser
    from src.eqprop.xyce_util.weightClipper import weightClipper
    from src.eqprop.xyce_util.xyce import XyceSim

log = get_pylogger(__name__)

NpOrTensor = np.ndarray | torch.Tensor


class AbstractStrategy(ABC):
    """Abstract class for different strategies to solve for the equilibrium point of the
    network."""

    def __init__(
        self,
        activation: Callable | str,
        max_iter: int = 30,
        atol: float = 1e-6,
        **kwargs,
    ):
        self.activation = activation
        self.max_iter = max_iter
        self.atol = atol
        self.dims = []  # hidden layer dimensions
        self.W = []
        self.B = []

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

    @abstractmethod
    def reset(self):
        """Reset the internal states after 1 iteration."""
        ...

    # TODO: 일부레이어만 bias 있다면? -> eqprop 모듈화
    def set_strategy_params(self, model: torch.nn.Module) -> None:
        """Set strategy parameters from nn.Module."""
        if not self.W and not self.B:
            for name, param in model.named_parameters():
                if name.endswith("weight"):
                    self.W.append(param)
                    self.dims.append(param.shape[0])
                elif name.endswith("bias"):
                    self.B.append(param)


class SPICEStrategy(AbstractStrategy):
    """Calculate Node potentials with SPICE."""

    """
    def __new__(cls):
        # check if ngspice is installed
        cls._check_spice()
        return super().__new__(cls)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def _check_spice(cls):
        raise NotImplementedError()


class PythonStrategy(AbstractStrategy):
    """Calculate Node potentials with Python."""

    def __init__(
        self,
        amp_factor: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        """Initialize the solver."""
        self.OTS = self.activation
        self.amp_factor = amp_factor
        self.attrchecked: bool = False
        self.free_solution: torch.Tensor | None = None
        # self.sparse = (
        #     True if sum([dim**2 for dim in self.dims]) / sum(self.dims) ** 2 < 0.1 else False
        # )

    @property
    def free_solution(self):
        return self._free_solution.detach().clone() if self._free_solution is not None else None

    @free_solution.setter
    def free_solution(self, value: torch.Tensor | None):
        self._free_solution = value.detach().clone() if value is not None else None

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


class SecondOrderStrategy(PythonStrategy):
    def __init__(self, add_nonlin_last: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._L = None
        self._R = None
        self.add_nonlin_last = add_nonlin_last
        self._set: bool = False

    @torch.no_grad()
    def laplacian(self) -> torch.Tensor:
        """Compute the 2D Laplacian matrix and cache it."""
        if self._L is None:
            dims = self.dims
            size = sum(dims)
            paddedG = [torch.zeros(dims[0], size).float()]
            [
                paddedG.append(F.pad(-g, (sum(dims[:i]), sum(dims[1 + i :]))))
                for i, g in enumerate(self.W[1:])
            ]
            Ll = torch.cat(paddedG, dim=-2)
            L = Ll + Ll.mT * self.amp_factor
            D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(self.W[0].sum(-1), (0, size - dims[0]))
            L += D0.diag()
            self._L = L
            self._set = True
        elif self._set is False:
            raise ValueError("Reset the strategy before calling")
        return self._L.detach().clone()

    @torch.no_grad()
    def jacobian(self, v: NpOrTensor) -> torch.Tensor:
        """Compute the 3D Jacobian of the residual L - a_r(v)"""
        L = self.laplacian()
        batchsize = v.size(0) if len(v.shape) == 2 else 1
        J = L.expand(batchsize, *L.shape).clone()
        if self.add_nonlin_last:
            J.diagonal(dim1=1, dim2=2)[:] -= self.OTS.a(v)
        else:
            J.diagonal(dim1=1, dim2=2)[:, : -self.dims[-1]] -= self.OTS.a(v[:, : -self.dims[-1]])
        if type(v) == np.ndarray:
            return J.numpy()
        elif type(v) == torch.Tensor:
            return J
        else:
            raise TypeError(f"unsupported type {type(v)} while constructing Jacobian")

    @torch.no_grad()
    def rhs(self, x) -> torch.Tensor:
        """Compute the 2D RHS vector without i_ext and cache it."""
        dims = self.dims
        size = sum(dims)
        if self._R is None:
            B = torch.zeros(size).type_as(x) if not self.B else torch.cat(self.B, dim=-1)
            B = B.expand(x.size(0), *B.shape).clone()
            B[:, : dims[0]] += x @ self.W[0].T
            B *= self.amp_factor
            self._R = B
        elif self._set is False:
            raise ValueError("Reset the strategy before calling")
        return self._R.detach().clone()

    @torch.no_grad()
    def residual(
        self,
        v: NpOrTensor,
        x: torch.Tensor,
        i_ext: torch.Tensor | None,
    ):
        """Compute the residual Lv - R - i_r(v)"""
        L = self.laplacian()
        R = self.rhs(x)
        if i_ext is not None:
            R[:, -self.dims[-1] :] += i_ext * self.amp_factor
        if type(v) == np.ndarray:
            v = torch.from_numpy(v).type_as(x)
        f = torch.einsum("...i, oi->...o", v, L) - R
        if self.add_nonlin_last:
            f -= self.OTS.i(v)
        else:
            f[:, : -self.dims[-1]] -= self.OTS.i(v[:, : -self.dims[-1]])
        if type(v) == torch.Tensor:
            return f
        elif type(v) == np.ndarray:
            return f.numpy()
        else:
            raise TypeError(f"unsupported type {type(v)} while constructing residual")

    @torch.no_grad()
    def lin_solve(self, x, i_ext) -> torch.Tensor:
        """Solve the linear system Lv = R + i_ext."""
        if self._free_solution is not None and i_ext is not None:
            v = self.free_solution
        else:
            lo, info = torch.linalg.cholesky_ex(self.laplacian())
            R = self.rhs(x)
            if i_ext is not None:
                R[-self.dims[-1] :] += i_ext * self.amp_factor
            v = torch.cholesky_solve(R.unsqueeze(-1), lo).squeeze(-1)
        return v

    def reset(self):
        self._L = None
        self._R = None
        self._set = False


class ScipyStrategy(SecondOrderStrategy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def solve(self, x: torch.Tensor, i_ext: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        if x.device != torch.device("cpu"):
            raise ValueError("ScipyStrategy only supports cpu.")
        self.check_and_set_attrs(kwargs)
        v_init = self.lin_solve(x, i_ext).numpy()
        vout_list = []
        for xi in x:
            vout = fsolve(
                self.residual,
                x0=v_init,
                args=(xi, i_ext),
                fprime=self.jacobian,
                xtol=self.atol,
                maxfev=self.max_iter,
            )
            vout_list.append(vout)
        vout = np.stack(vout_list, axis=0)
        vout = torch.from_numpy(vout).type_as(x)
        if i_ext is None:
            self._free_solution = vout.detach().clone()
        nodes = list(vout.split(self.dims, dim=1))
        return nodes


class XYCEStrategy(SPICEStrategy):
    """Calculate Node potentials with Xyce."""

    @classmethod
    def _check_spice(cls):
        if _XYCE_AVAILABLE:
            pass
        else:
            raise ImportError("Xyce is not installed.")

    # TODO: set up xyce configuration, generate netlist
    def __init__(self, diode_path, diode_name, L, U, beta, **kwargs) -> None:
        super().__init__(**kwargs)

        """Create pyspice circuit instance using dictionary type parameters.

        spice_params = {
          "L" : lower limit of weight of each layer<List/float>,
          "U" : upper limit of weight of each layer<List/float>,
          "A" : amplitude of bidirectional amplifier<float/int>
          "Diode" : {
            "Path" : path to diode spice model file<string>,
            "ModelName" : name of diode model<string>,
            "Rectifier" : rectifying subcircuit instance <SubcircuitFactory>
                }
          "alpha" : learning rate of each layer <List/float>
        "beta" : weight for Cost function <Float>
        }
        """
        self.beta = beta
        self.amp_factor = 1.0
        self.attrchecked: bool = False
        self.mpi_commands = ["mpirun", "-use-hwthread-cpus", "-np", "1", "-cpu-set"]

        self.SPICE_params = {
            "A": self.amp_factor,
            "beta": self.beta,
            "L": L,
            "U": U,
            "Diode": {
                "Path": diode_path,
                "ModelName": diode_name,
                "Rectifier": "BidRectifier",
            },
        }

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
        for attr in ["dims"]:
            if getattr(self, attr) is None:
                raise ValueError(f"{attr} must be set before calling")

    # TODO: return nodes in equilibrium
    def solve(self, x: torch.Tensor, i_ext, **kwargs) -> list[torch.Tensor]:
        # get params
        (W, B) = kwargs.get("params", (self.W, self.B))
        self.check_and_set_attrs(kwargs)

        # generate netlist, concat W & B

        weight_list = {}
        if len(B) != 0:
            bias = B[0].view(len(W[0]), 1)
            weight_list[0] = torch.cat((W[0], bias), dim=1)
            weight_list[1] = W[1]

        else:
            weight_list[0] = W[0]
            weight_list[1] = W[1]

        dims = [
            len(weight_list[0][0]),
            len(weight_list[0]),
            len(weight_list[1].transpose(0, 1)[0]),
        ]

        """
        lst = [1 for _ in range(len(dims)-1)]
        lst.append(0)
        dims = [x + y for x, y in zip(dims, lst)]
        """
        self.Pycircuit = _createCircuit(
            weight_list, dims, **self.SPICE_params
        )  # create circuit with bias
        self.circuit = MyCircuit.copyFromCircuit(self.Pycircuit)
        self.clipper = weightClipper(L=self.SPICE_params["L"], U=None)
        circuit = self.circuit

        batch_size = x.size(0)
        nodes = []
        # set mpi_commands
        if batch_size == 1:
            self.mpi_commands.pop()
            self.mpi_commands.pop()
            self.mpi_commands.append(str(os.cpu_count() - 2))
            self.mpi_commands.append("-cpu-set")
            self.mpi_commands.append("2-" + str(os.cpu_count() - 1))

        for i in range(len(dims) - 1):
            nodes.append(None)
            nodes[i] = torch.zeros(batch_size, dims[len(dims) - i - 1])
        for i in range(batch_size):
            SPICEParser.clampLayer(circuit, x[i])
            xyce = XyceSim(mpi_command=None)
            if i_ext is None:
                self._free_solution = None
                raw_file = xyce(spice_input=circuit)
                vout = SPICEParser.fastRawfileParser(
                    raw_file, nodenames=circuit.nodes, dimensions=dims
                )

            if i_ext is not None:
                self._free_solution = None
                SPICEParser.releaseLayer(circuit, i_ext[i])
                raw_file = xyce(spice_input=circuit)
                vout = SPICEParser.fastRawfileParser(
                    raw_file, nodenames=circuit.nodes, dimensions=dims
                )
            for j in range(len(dims) - 1):
                Vlist = torch.tensor(vout[1][len(dims) - j - 2])
                nodes[j][i] = Vlist

        temp = nodes[0]
        nodes[0] = nodes[1]
        nodes[1] = temp

        return nodes

    def process_data(i, x, circuit, nodes, j):
        SPICEParser.clampLayer(circuit, x[i])
        xyce = XyceSim(mpi_command=None)
        vout = xyce(x[i])
        nodes[j][i] = vout
        return i, vout


class PinvStrategy(SecondOrderStrategy):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @torch.no_grad()
    def solve(self, x, i_ext, **kwargs) -> list[torch.Tensor]:
        batchsize = x.size(0)
        self.check_and_set_attrs(kwargs)
        L = self.laplacian()
        R = self.rhs(x)
        if i_ext is not None:
            R[-self.dims[-1] :] += i_ext * self.amp_factor
        # initial solution
        lo, info = torch.linalg.cholesky_ex(L)
        vout = torch.cholesky_solve(R.unsqueeze(-1), lo).squeeze(-1)
        # add contribution from nonlinearity
        vout -= (
            torch.pinverse(L).expand(batchsize, -1, -1) @ self.OTS.i(vout).unsqueeze(-1)
        ).squeeze()
        nodes = list(vout.split(self.dims, dim=1))
        return nodes


class NewtonStrategy(SecondOrderStrategy):
    r"""Solve J\Delta{X}=-f with Newton's method."""

    def __init__(self, clip_threshold, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_threshold = clip_threshold

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
        self.check_and_set_attrs(kwargs)
        if i_ext is None:
            self.free_solution = None

        if type(self.OTS) == eqprop_util.SymOTS:
            vout = self._densecholsol2(x, i_ext)
        else:
            vout = self._densecholsol(x, i_ext)
        if i_ext is None:
            self.free_solution = vout
        nodes = list(vout.split(self.dims, dim=1))
        return nodes

    @torch.no_grad()
    def _densecholsol(
        self,
        x: torch.Tensor,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J\Delta{X}=-f with dense cholesky/LU decomposition.

        Args:
            x (torch.Tensor): Input.
            W (list): List of weight matrices. Each element of the list size (dim, dim).
            B (list, optional): List of bias vectors. Defaults to None. Each element of the list size (batchsize, dim).
            i_ext ([type], optional): External current. Defaults to None.
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_util.self.OTS().
            self.max_iter (int, optional): Maximum number of iterations. Defaults to 30.
            self.atol (float, optional): Absolute tolerance. Defaults to 1e-6.
            self.amp_factor (float, optional): Layerwise voltage&current amplitude factor. Defaults to 1.0.
        """
        v = self.lin_solve(x, i_ext)
        residual_v = torch.ones(1).type_as(x)
        idx = 1
        while (residual_v.abs().max() > self.atol) and (idx < self.max_iter):
            residual_v = self.residual(v, x, i_ext).unsqueeze(-1)
            J = self.jacobian(v)
            # or SPOSV
            if self.amp_factor == 1.0:
                lo, info = torch.linalg.cholesky_ex(J)
                if any(info):  # singular
                    log.debug(f"J is singular, info={info}")
                    lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
                    dv = torch.linalg.lu_solve(lo, piv, -residual_v).squeeze(-1)
                else:
                    dv = torch.cholesky_solve(-residual_v, lo).squeeze(-1)
            else:
                lo, piv, info = torch.linalg.lu_factor_ex(J)
                if any(info):
                    lo, piv, info = torch.linalg.lu_factor_ex(J + 1e-6 * torch.eye(J.size(-1)))
                    dv = torch.linalg.lu_solve(lo, piv, -residual_v).squeeze(-1)
                else:
                    dv = torch.linalg.lu_solve(lo, piv, -residual_v).squeeze(-1)
            # limit the voltage change
            dv = dv.clamp(min=-self.clip_threshold, max=self.clip_threshold)
            idx += 1
            v += dv

        log.debug(f"condition number of J: {torch.linalg.cond(J[0]):.2f}")
        if idx == self.max_iter:
            log.warning(
                f"stepsolve did not converge in {self.max_iter} iterations, residual={residual_v.abs().max():.3e}"
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
            v = self.free_solution
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
        L = self.laplacian()
        R = self.rhs(x)
        if i_ext is not None:
            R[-dims[-1] :] += i_ext * self.amp_factor

        v = self.lin_solve(x, i_ext)
        residual = torch.ones(1).type_as(x)
        L = L.expand(batchsize, *L.shape)
        idx = 1
        while (residual.abs().max() > self.atol) and (idx < self.max_iter):
            # nonlinearity comes here
            # log.debug(f"absvmax: {v.abs().max():.3e}, absvmin: {v.abs().min():.3e}")
            a_inv = (1 / self.OTS.a(v)).unsqueeze(-1)
            J = a_inv * (L.clone()) + torch.eye(L.size(-1))  # L shape, a_inv shape?
            residual = torch.bmm(L, v.unsqueeze(-1)) - R.clone().unsqueeze(-1)
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
            v = self.free_solution
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


class LMStrategy(PythonStrategy):
    r"""Solve J\Delta{X}=-f with Levenberg-Marquardt method."""

    def __init__(self, clip_threshold, lambda_factor, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clip_threshold = clip_threshold
        self.lambda_factor = lambda_factor

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
        (W, B) = kwargs.get("params", (self.W, self.B))
        if i_ext is None:
            self.free_solution = None
        vout = self._LMdensecholsol(x, W, B, i_ext)
        if i_ext is None:
            self.free_solution = vout
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
            v = self.free_solution
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
