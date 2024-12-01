from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
import time

from scipy.optimize import minimize
import numpy as np
import torch

from src.prog_scheme.types import (
    BatchedInput,
    BatchedOutput,
    StateVec,
    TorchDeviceArray,
    TorchBatchedVecPair,
)


class AbstractController(ABC):
    """Abstract class for controllers."""

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, x) -> Any:
        """Return the control signal."""
        pass


class ComposedController(AbstractController):
    """Composition of controllers."""

    def __init__(self, *controllers: AbstractController):
        self.controllers = controllers

    def __call__(self, x):
        u = x
        for controller in self.controllers:
            u = controller(u)
        return u


class LQRController(AbstractController):
    def __init__(self, Q, R):
        super().__init__()
        self.Q = Q
        self.R = R

    def __call__(self, x):
        def cost(u):
            return x @ self.Q @ x + u @ self.R @ u

        u0 = 0
        res = minimize(cost, u0)
        return res.x


class BaseDeviceController(AbstractController):
    """Base class for device programming conrtollers."""

    def __init__(self, target_w: TorchDeviceArray, batch_size: int):
        super().__init__()
        self.set_target(target_w)
        self.batch_size = batch_size

    def __call__(self, x) -> tuple[BatchedInput, BatchedOutput]:  # noqa F722
        """Return the pair of (batched) update vectors"""
        ...

    def set_target(self, target_w: TorchDeviceArray):
        self._target_w = target_w
        self.input_size = target_w.shape[1]
        self.output_size = target_w.shape[0]

    def reset(self):
        """Reset the controller."""
        pass


class RowController(BaseDeviceController):
    """Controller for programming a (batched) row(s)."""

    def __init__(self, target_w: TorchDeviceArray, batch_size: int):
        super().__init__(target_w, batch_size)
        self._rows_selected = torch.arange(batch_size)

    def __call__(self, x: TorchDeviceArray) -> TorchBatchedVecPair:  # noqa F722
        u_rows = self._target_w[self._rows_selected, :] - x[self._rows_selected, :]
        u_columns = torch.zeros((self.batch_size, self.output_size))
        u_columns[torch.arange(self.batch_size), self._rows_selected] = 1
        self._rows_selected = (self._rows_selected + self.batch_size) % self.output_size
        return (u_columns, u_rows)

    def reset(self):
        self._rows_selected = torch.arange(self.batch_size)


class ColumnController(RowController):
    """Controller for programming a (batched) column(s)."""

    def __init__(self, target_w: TorchDeviceArray, batch_size: int):
        super().__init__(target_w.T, batch_size)


class SVDController(BaseDeviceController):
    """Controller for programming using SVD."""

    def __init__(self, target_w: TorchDeviceArray, batch_size: int, svd_every_k_iter: int):
        super().__init__(target_w, batch_size)
        self.svd_every_k_iter = svd_every_k_iter
        self._idx = 0

        assert batch_size * svd_every_k_iter <= min(self.input_size, self.output_size), (
            "batch_size * svd_every_k_iter should be less than or equal to the minimum of "
            "input_size and output_size."
        )

    def __call__(self, x: TorchDeviceArray) -> TorchBatchedVecPair:  # noqa F722
        """Perform SVD every k iterations. and return the top batch_size singular vectors."""
        if (i := self._idx % self.svd_every_k_iter) == 0:
            dW = self._target_w - x
            self._U, self._S, self._Vh = torch.linalg.svd(dW, full_matrices=False)
        s = self._S[i * self.batch_size : (i + 1) * self.batch_size]
        sqrt_s = torch.sqrt(s).unsqueeze(1)
        u = self._U[:, i * self.batch_size : (i + 1) * self.batch_size].T * sqrt_s
        v = self._Vh[i * self.batch_size : (i + 1) * self.batch_size, :] * sqrt_s
        self._idx += 1
        return (v, u)

    def reset(self):
        self._idx = 0
        self._S = None
        self._U = None
        self._Vh = None


class RobustSVDController(SVDController):
    """Controller for programming using rank minimization."""

    def __init__(
        self,
        target_w: TorchDeviceArray,
        batch_size: int,
        svd_every_k_iter: int,
        alpha: float,
        f: Callable,
    ):
        super().__init__(target_w, batch_size, svd_every_k_iter)
        self.alpha = alpha
        self.f = f
        self._min_rank = batch_size * svd_every_k_iter

    def __call__(self, x: TorchDeviceArray) -> TorchBatchedVecPair:  # noqa F722
        """Perform robust SVD every k iterations. and return the top batch_size singular vectors."""
        if i := (self._idx % self.svd_every_k_iter) == 0:
            # start = time.time()
            x = x.numpy().flatten().astype(np.float64)
            target = self._target_w.numpy().flatten().astype(np.float64)
            dW = target - x
            cost_args = (
                x,
                target,
                (self.output_size, self.input_size),
                self.alpha,
                self.f,
            )
            dW = minimize(self.cost_fn, x0=dW, args=cost_args).x.reshape(self._target_w.shape)
            # print(f"Time taken: {time.time() - start}")
            dW_ = torch.from_numpy(dW)
            self._U, self._S, self._V = torch.linalg.svd(dW_, full_matrices=False)
        s = self._S[i * self.batch_size : (i + 1) * self.batch_size]
        sqrt_s = torch.sqrt(s).unsqueeze(1)
        u = self._U[:, i * self.batch_size : (i + 1) * self.batch_size].T * sqrt_s
        v = self._V[i * self.batch_size : (i + 1) * self.batch_size, :] * sqrt_s
        self._idx += 1
        return (v, u)

    @staticmethod
    def cost_fn(
        u: StateVec,
        x: StateVec,
        target: StateVec,
        mtx_size: tuple[int, int],
        alpha: float,
        f: Callable[[StateVec, StateVec], StateVec],
    ):
        dW = (target - f(x, u)).reshape(*mtx_size)
        return np.linalg.norm(dW, ord="fro") + alpha * np.linalg.norm(  # codespell:ignore fro
            dW, ord="nuc"
        )
