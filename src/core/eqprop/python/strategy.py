from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.utils import _QPSOLVERS_AVAILABLE, _SCIPY_AVAILABLE, _SPICE_AVAILABLE, RankedLogger

if _SCIPY_AVAILABLE:
    from scipy.optimize import fsolve

if _QPSOLVERS_AVAILABLE:
    import proxsuite
    from qpsolvers import solve_qp

if _SPICE_AVAILABLE:
    from src.core.spice import circuits, xyce, spice_utils

from src.core.eqprop.python import activation

log = RankedLogger(__name__, rank_zero_only=True)

NpOrTensor = np.ndarray | torch.Tensor

__all__ = [
    "AbstractStrategy",
    "ScipyStrategy",
    "XyceStrategy",
    "NewtonStrategy",
    "GradientDescentStrategy",
    "QPStrategy",
    "ProxQPStrategy",
    "FirstOrderStrategy",
]


def cache_free_solution(func: callable):
    """Cache the free-phase solution of the network."""

    def wrapper(self, x, i_ext, **kwargs):
        if i_ext is None:
            self.free_solution = None
        vout = func(self, x, i_ext, **kwargs)
        if i_ext is None:
            self.free_solution = vout
        return vout

    return wrapper


class AbstractStrategy(ABC):
    """Abstract class for different strategies to solve for the equilibrium point of the network.

    Args:
        activation (Callable | str): activation function of the network.
        max_iter (int): maximum number of iterations.
        atol (float): absolute tolerance.

    """

    def __init__(
        self,
        activation: activation.AbstractRectifier,
        max_iter: int = 30,
        atol: float = 1e-6,
        **kwargs,
    ):
        self.activation = activation
        self.max_iter = max_iter
        self.atol = atol
        # hidden layer dimensions, set by the solver.set_model() method
        self.dims = []
        self.W = []
        self.B = []

    @abstractmethod
    def solve(self, x, i_ext, **kwargs) -> torch.Tensor:
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
        """Reset the internal states at the beginning of 1 iteration."""
        ...

    def set_bias_type(self, bias_type: str) -> None:
        """Set the bias type of the model."""
        self.bias_type = bias_type
        ...


class AbstractSPICEStrategy(AbstractStrategy):
    """Calculate Node potentials with SPICE."""

    """
    def __new__(cls):
        # check if ngspice is installed
        cls._check_spice()
        return super().__new__(cls)
    """

    def __init__(self, SPICE_params: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.SPICE_params = SPICE_params

    @classmethod
    def _check_spice(cls):
        """Check if spice is installed."""
        raise NotImplementedError()


class XyceStrategy(AbstractSPICEStrategy):
    """Get Node potentials with Xyce."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mpi_commands = kwargs.get("mpi_commands") or ["mpirun", "-use-hwthread-cpus"]

        if self.mpi_commands[-1] == "-cpu-set":
            # self.mpi_commands.append(str(id + 1))
            pass
        self.sim = xyce.XyceSim(mpi_commands=self.mpi_commands)

    def solve(self, x, i_ext, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network with Xyce."""
        if i_ext is None:
            self.create_netlist(x)
            spice_utils.SPICENNParser.updateWeight(self.circuit, self.W)
        nodes_list = []
        # not multiprocessing yet
        batch_size = x.size(0)
        dims = [len(x[0])] + self.dims
        for i in range(batch_size):
            if i_ext is None:
                spice_utils.SPICENNParser.clampLayer(self.circuit, x[i])
            else:
                spice_utils.SPICENNParser.releaseLayer(self.circuit, -i_ext)

            raw_file = self.sim(spice_input=self.circuit)
            voltages = spice_utils.SPICENNParser.fastRawfileParser(
                raw_file, nodenames=self.circuit.nodes, dimensions=dims
            )

            combined_voltages = np.concatenate([voltages[1][0], voltages[1][1]])
            nodes_list.append(combined_voltages)

        nodes_array = np.stack(nodes_list, axis=0)
        nodes = torch.from_numpy(nodes_array).type_as(x)
        return nodes

    def create_netlist(self, x):
        """Convert input to netlist."""
        """self.W, self.B, self.dims / diode model name in self.SPICE_params."""
        if self.circuit is None:
            self.Pycircuit = circuits.create_circuit(
                input=x, bias=self.B, W=self.W, dimensions=self.dims, **self.SPICE_params
            )
            self.circuit = circuits.ShallowCircuit.copyFromCircuit(self.Pycircuit)
            del self.Pycircuit

    def reset(self):
        """Reset the internal states after 1 iteration."""
        return NotImplementedError()


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
        if kwargs is None and self.attrchecked:
            pass
        else:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    log.warning(f"key {key} not found in {self.__class__.__name__}")
            for attr in ["OTS", "dims", "W"]:
                if getattr(self, attr) is None:
                    raise ValueError(f"{attr} must be set before calling")


class FirstOrderStrategy(PythonStrategy):
    """Solve for the equilibrium point of the network with first order approximation."""

    def __init__(self, add_nonlin_last: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._L = None
        self._R = None
        self._B = None
        self.add_nonlin_last = add_nonlin_last
        self._set: bool = False

    @torch.no_grad()
    def bias(self) -> torch.Tensor:
        """Compute the 1D Bias row vector and cache it."""
        if self._B is None:
            dims = self.dims
            size = sum(dims)
            B = torch.zeros(size).type_as(self.W[0]) if not self.B else torch.cat(self.B, dim=-1)
            assert len(B.shape) == 1
            self._B = B
        return self._B.detach().clone()

    @torch.no_grad()
    def laplacian(self) -> torch.Tensor:
        """Compute the 2D Laplacian + bias matrix and cache it.

        Returns:
            torch.Tensor: (size, size)

        """
        if self._L is None:
            dims = self.dims
            size = sum(dims)
            paddedG = [torch.zeros(dims[0], size).type_as(self.W[0])]
            [
                paddedG.append(F.pad(-g, (sum(dims[:i]), sum(dims[1 + i :]))))
                for i, g in enumerate(self.W[1:])
            ]
            Ll = torch.cat(paddedG, dim=-2)
            L = Ll + Ll.mT * self.amp_factor
            D0 = -Ll.sum(-2) - Ll.sum(-1) + F.pad(self.W[0].sum(-1), (0, size - dims[0]))
            L += D0.diag()
            L += self.bias().diag() if self.B else 0
            self._L = L
            self._set = True
        elif self._set is False:
            raise ValueError("Reset the strategy before calling")
        return self._L.detach().clone()

    @torch.no_grad()
    def rhs(self, x) -> torch.Tensor:
        """Compute the batched 2D RHS vector {-bias +-[x@W0.T,0]} without i_ext and cache it.

        Args:
            x (torch.Tensor): input of the network. (batchsize, size)

        Returns:
            torch.Tensor: batched 2D RHS vector. (batchsize, size)

        """
        dims = self.dims
        if self._R is None:
            B = -self.bias()  # 1D row vector
            if len(x.shape) == 2:  # batched
                B = B.expand(x.size(0), *B.shape).clone()
            elif len(x.shape) == 1:
                B.unsqueeze_(0)
                x = x.unsqueeze(0)
            else:
                raise ValueError(f"unsupported shape {x.shape} while constructing RHS")
            B[:, : dims[0]] -= x @ self.W[0].T
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
        """Compute the residual v(L+b) + R + i_r(v) where v, R, i_r(v) are batched vectors.

        Args:
            v (NpOrTensor): node potentials. (batchsize, size)
            x (torch.Tensor): input of the network. (batchsize, size)
            i_ext (torch.Tensor): external current. (batchsize, size)

        Returns:
            torch.Tensor: residual vector. (batchsize, size)

        """
        L = self.laplacian()
        R = self.rhs(x)
        if i_ext is not None:
            R[:, -self.dims[-1] :] += i_ext * self.amp_factor
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type_as(x)
        if len(v.shape) == 1:
            v = v.unsqueeze(0)
        f = torch.einsum("bi, oi->bo", v, L) + R
        if self.add_nonlin_last:
            f += self.OTS.i(v)
        else:
            f[:, : -self.dims[-1]] += self.OTS.i(v[:, : -self.dims[-1]])
        if isinstance(v, torch.Tensor):
            return f
        elif isinstance(v, np.ndarray):
            return f.numpy()
        else:
            raise TypeError(f"unsupported type {type(v)} while constructing residual")

    @torch.no_grad()
    def lin_solve(self, x, i_ext) -> torch.Tensor:
        """Solve the linear system (L+b)v = -(R + i_ext)."""
        if self._free_solution is not None and i_ext is not None:
            v = self.free_solution
        else:
            lo = torch.linalg.cholesky(self.laplacian())
            R = self.rhs(x)
            if i_ext is not None:
                R[:, -self.dims[-1] :] += i_ext * self.amp_factor
            v = torch.cholesky_solve(-R.unsqueeze(-1), lo).squeeze(-1)
        return v

    def reset(self):
        """Reset cache for every free phase."""
        self._L = None
        self._R = None
        self._B = None
        self._set = False


class SecondOrderStrategy(FirstOrderStrategy):
    """Solve for the equilibrium point of the network with second order approximation.

    Args:
        eps (float): small value to add to the diagonal of the Laplacian matrix.

    """

    def __init__(self, eps: float = 1e-8, **kwargs) -> None:
        super().__init__(**kwargs)
        self.eps = eps

    @torch.no_grad()
    def jacobian(self, v: NpOrTensor) -> torch.Tensor:
        """Compute the 3D Jacobian of the residual L + b + a_r(v)"""
        L = self.laplacian() + self.eps * torch.eye(v.size(-1)).type_as(v)
        if len(v.shape) == 2:
            batchsize = v.size(0)
        elif len(v.shape) == 1:
            batchsize = 1
            v = v.unsqueeze(0)
        else:
            raise ValueError(f"unsupported shape {v.shape} while constructing Jacobian")
        J = L.expand(batchsize, *L.shape).clone()
        if self.add_nonlin_last:
            J.diagonal(dim1=1, dim2=2)[:] += self.OTS.a(v)
        else:
            J.diagonal(dim1=1, dim2=2)[:, : -self.dims[-1]] += self.OTS.a(v[:, : -self.dims[-1]])
        if isinstance(v, np.ndarray):
            return J.numpy()
        elif isinstance(v, torch.Tensor):
            return J
        else:
            raise TypeError(f"unsupported type {type(v)} while constructing Jacobian")


class ScipyStrategy(SecondOrderStrategy):
    """Solve for the equilibrium point of the network with Scipy."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def solve(self, x: torch.Tensor, i_ext: torch.Tensor, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network.

        Currently only supports cpu. & does not support batched operation in parallel.
        """
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
        return vout


class GradientDescentStrategy(FirstOrderStrategy):
    """Solve for the equilibrium point of the network with gradient descent.

    Args:
        alpha (float): attenuation rate.

    """

    def __init__(self, alpha: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha

    @cache_free_solution
    @torch.no_grad()
    def solve(self, x, i_ext, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network with gradient descent."""
        self.check_and_set_attrs(kwargs)
        v = self.lin_solve(x, i_ext)
        for idx in range(self.max_iter):
            dv = -self.residual(v, x, i_ext)
            v += self.alpha * dv
            if dv.abs().max() < self.atol:
                log.debug(f"stepsolve converged in {idx} iterations")
                return v
        log.warning(
            f"stepsolve did not converge in {self.max_iter} iterations, residual={dv.abs().max():.3e}"
        )
        return v


class QPStrategy(FirstOrderStrategy):
    def __init__(self, add_nonlin_last: bool = True, solver_type: str = "proxqp", **kwargs) -> None:
        """Solve for the equilibrium point of the network with qpsolvers library."""
        super().__init__(add_nonlin_last, **kwargs)
        self.solver_type = solver_type

    @torch.no_grad()
    def solve(self, x, i_ext, **kwargs) -> torch.Tensor:
        self.check_and_set_attrs(kwargs)
        P = self.laplacian()
        R = self.rhs(x)
        if i_ext is not None:
            R[:, -self.dims[-1] :] += i_ext * self.amp_factor
        q = R.squeeze()
        lb = self.OTS.Vl * np.ones_like(q)
        ub = self.OTS.Vr * np.ones_like(q)
        v = solve_qp(P, q, lb=lb, ub=ub, solver=self.solver_type, **kwargs)
        v = torch.from_numpy(v).type_as(x).unsqueeze(0)
        # v_lin = self.lin_solve(x, i_ext)
        # sgn = torch.sign(v_lin-v)
        # v_logdiff = 0.02*(v_lin-v).abs().log1p()
        # v += sgn*v_logdiff
        return v

    def sparse_laplacian(self):
        """Compute the 2D Laplacian + bias matrix in bsr format and cache it.

        Raises:
            ValueError: _description_
            RuntimeError: _description_
            NotImplementedError: _description_

        Returns:
            _type_: _description_

        """


class ProxQPStrategy(QPStrategy):
    """Solve for the equilibrium point of the network with ProxQP."""

    def __init__(self, num_threads: int | None, **kwargs) -> None:
        super().__init__(solver_type="proxqp", **kwargs)
        self.num_threads = (
            proxsuite.proxqp.omp_get_max_threads() - 1 if num_threads is None else num_threads
        )

    @torch.no_grad()
    def solve(self, x, i_ext, **kwargs) -> torch.Tensor:
        batch_size, _ = x.shape
        g = self.rhs(x).cpu().numpy()
        if i_ext is None:
            H = self.laplacian().cpu().numpy()
            n = n_ineq = H.shape[0]
            A = b = C = lower = upper = None
            self.lb = self.OTS.Vl * np.ones(n)
            self.ub = self.OTS.Vr * np.ones(n)
            self.qps = proxsuite.proxqp.dense.VectorQP()
            for i in range(batch_size):
                qp = proxsuite.proxqp.dense.QP(n, 0, n_ineq, True)
                qp.init(H, g[i], A, b, C, lower, upper, self.lb, self.ub)
                self.qps.append(qp)
        else:
            g[:, -self.dims[-1] :] += i_ext.cpu().numpy()
            for idx, qp in enumerate(self.qps):
                qp.settings.initial_guess = (
                    proxsuite.proxqp.InitialGuess.WARM_START_WITH_PREVIOUS_RESULT
                )
                qp.update(g=g[idx], l_box=self.lb, u_box=self.ub)

        proxsuite.proxqp.dense.solve_in_parallel(self.qps, self.num_threads)
        nodes_list = []
        for i in range(batch_size):
            vout = self.qps[i].results.x
            nodes_list.append(torch.from_numpy(vout).type_as(x))
        nodes = torch.stack(nodes_list, dim=0)
        return nodes

    def reset(self):
        super().reset()
        # del self.qps


class NewtonStrategy(SecondOrderStrategy):
    r"""Solve J\Delta{X}=-f with Newton's method.

    Args:
        clip_threshold (float): threshold for voltage change.
        attn_factor (float): attenuation factor for voltage change.

    """

    def __init__(
        self, clip_threshold, attn_factor: float = 1, momentum: float = 0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.clip_threshold = clip_threshold
        self.attn_factor = attn_factor
        self.momentum = momentum

    @cache_free_solution
    @torch.no_grad()
    def solve(self, x: torch.Tensor, i_ext, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network.

        Args:
            x (torch.Tensor): input of the network.
            i_ext (torch.Tensor): external current.
        KwArgs:
            params (tuple[list[torch.Tensor], list[torch.Tensor]]): weights and biases of the model.
            OTS (eqprop_utils.P3OTS): nonlinearity.
            dims (list): dimensions of the model.
            max_iter (int): maximum number of iterations.
            atol (float): absolute tolerance.
            amp_factor (float): inter-layer potential amplifying factor.

        """
        self.check_and_set_attrs(kwargs)
        if isinstance(self.OTS, activation.SymOTS):
            vout = self._densecholsol2(x, i_ext)
        else:
            vout = self._densecholsol(x, i_ext)
        return vout

    @torch.no_grad()
    def _densecholsol(
        self,
        x: torch.Tensor,
        i_ext=None,
    ) -> torch.Tensor:
        r"""Solve J\Delta{X}=-f with dense cholesky/LU decomposition. uses adaptive step size.

        Args:
            x (torch.Tensor): Input.
            W (list): List of weight matrices. Each element of the list size (dim, dim).
            B (list, optional): List of bias vectors. Defaults to None. Each element of the list size (batchsize, dim).
            i_ext ([type], optional): External current. Defaults to None.
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_utils.self.OTS().
            self.max_iter (int, optional): Maximum number of iterations. Defaults to 30.
            self.atol (float, optional): Absolute tolerance. Defaults to 1e-6.
            self.amp_factor (float, optional): Layerwise voltage&current amplitude factor. Defaults to 1.0.

        """
        v = self.lin_solve(x, i_ext)
        residual_v = self.residual(v, x, i_ext).unsqueeze(-1)
        J = None
        idx = 1
        p = torch.zeros_like(v)
        while (residual_v.abs().max() > self.atol) and (idx < self.max_iter):
            J = self.jacobian(v)
            # or SPOSV
            if self.amp_factor == 1.0:
                lo, info = torch.linalg.cholesky_ex(J)
                dv = torch.cholesky_solve(-residual_v, lo).squeeze(-1)
            else:
                lo, piv, info = torch.linalg.lu_factor_ex(J)
                dv = torch.linalg.lu_solve(lo, piv, -residual_v).squeeze(-1)
            # limit the voltage change
            if torch.isnan(dv).any():
                raise ValueError("dv contains NaN")
            dv.clamp_(min=-self.clip_threshold, max=self.clip_threshold)
            p = p * self.momentum + self.attn_factor * dv
            v_new = v + p
            residual_v_new = self.residual(v_new, x, i_ext).unsqueeze(-1)  # (batchsize, size, 1)
            if torch.any(residual_v_new.norm() > residual_v.norm()):
                log.debug(f"residual increased, idx={idx}")
                self.attn_factor *= 0.25
            if idx % 10 == 1:
                eigvals = torch.linalg.eigvals(self.jacobian(v))
                pd = torch.all(eigvals.real > 0) & torch.all(eigvals.imag == 0)
                log.debug(f"residual: {dv.abs().max():.3e}, PDness: {pd}")
            else:
                v = v_new
                residual_v = residual_v_new
                self.attn_factor *= 1.1
            idx += 1
        (
            log.debug(f"condition number of J: {torch.linalg.cond(J[0]):.2f}")
            if J is not None
            else None
        )
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
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_utils.self.OTS().
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
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_utils.self.OTS().
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
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_utils.self.OTS().
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
    def solve(self, x, i_ext, **kwargs) -> torch.Tensor:
        """Solve for the equilibrium point of the network.

        Args:
            x (torch.Tensor): input of the network.
            i_ext (torch.Tensor): external current.
        KwArgs:
            params (tuple[list[torch.Tensor], list[torch.Tensor]]): weights and biases of the model.
            OTS (eqprop_utils.P3OTS): nonlinearity.
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
        return vout

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
            self.OTS ([type], optional): Nonlinearity. Defaults to eqprop_utils.self.OTS().
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
            if torch.all(torch.norm(new_residuals, dim=(1, 2)) < torch.norm(residuals, dim=(1, 2))):
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
    """Compute the inverse of the Laplacian matrix."""
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
