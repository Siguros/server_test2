from abc import ABC, abstractmethod
from typing import Any

from jaxtyping import Float
from scipy.optimize import minimize
import numpy as np

DeviceArray = Float[np.ndarray, "output*input"]


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

    def __init__(self, target_w: DeviceArray):
        super().__init__()
        self.target_w = target_w

    def __call__(self, x) -> tuple[Float[np.ndarray, "batch in"], Float[np.ndarray, "batch out"]]:  # noqa F722
        """Return the pair of (batched) update vectors"""
        ...


class OneRowController(BaseDeviceController):
    """Controller for programming a single row."""

    def __init__(self, target_w: DeviceArray):
        super().__init__(target_w)
        self.row = 0
        self.input_size = target_w.shape[1]
        self.output_size = target_w.shape[0]

    def __call__(self, x) -> tuple[Float[np.ndarray, "batch in"], Float[np.ndarray, "batch out"]]:  # noqa F722
        # x is the current weight
        # target_w is the target weight
        # return the update vector
        u_row = self.target_w[self.row, :] - x[self.row, :]
        u_column = np.zeros(self.output_size)
        u_column[self.row] = 1
        self.row = (self.row + 1) % self.output_size
        return (u_column, u_row)


class BatchRowController(BaseDeviceController):
    """Controller for programming a batch of rows."""

    def __init__(self, target_w: DeviceArray, batch_size: int):
        super().__init__(target_w)
        self.rows = np.arange(batch_size)
        self.input_size = target_w.shape[1]
        self.output_size = target_w.shape[0]
        self.batch_size = batch_size

    def __call__(self, x) -> tuple[Float[np.ndarray, "batch in"], Float[np.ndarray, "batch out"]]:  # noqa F722
        u_rows = self.target_w[self.rows, :] - x[self.rows, :]
        u_columns = np.zeros((self.batch_size, self.output_size))
        u_columns[np.arange(self.batch_size), self.rows] = 1
        self.rows = (self.rows + self.batch_size) % self.output_size
        return (u_columns, u_rows)


class OneColumnController(OneRowController):
    """Controller for programming a single column."""

    def __init__(self, target_w: DeviceArray):
        super().__init__(target_w.T)
