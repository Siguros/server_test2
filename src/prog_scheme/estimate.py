from abc import ABC, abstractmethod

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from scipy import integrate


class BaseDeviceKF:
    """Base class for device programming with Kalman filter.

    x_k = x_{k-1} + u_k + w_k z_k = x_k + v_k
    """

    def __init__(self, dim: int, read_noise_std: float, update_noise_std: float):
        self.dim = dim
        self.q = read_noise_std**2
        self.r = update_noise_std**2
        self.P_scale = 1
        # Solve Riccati equation for S_scale
        # poly = np.array([1, -(self.r), -(self.r) * (self.q)])
        # self.S_scale = np.roots(poly).max()
        # self.K_scale = self.S_scale / (self.S_scale + q)
        self.x_est = np.zeros(dim)

    def predict(self, u_vec: np.ndarray):
        """Predict the next state of the device."""
        self.x_est += u_vec
        self.P_scale += self.q

    def update(self, z: np.ndarray):
        """Update the state estimation of the device after reading the device state."""
        y_res = z - self.x_est
        self.S_scale = self.P_scale + self.r
        self.K_scale = self.P_scale / self.S_scale
        self.x_est += self.K_scale * y_res
        self.P_scale = (1 - self.K_scale) * self.P_scale


class BaseDeviceEKF(ExtendedKalmanFilter, ABC):
    """EKF for device programming."""

    def __init__(self, dim, **kwargs):
        super().__init__(dim_x=dim, dim_z=dim, dim_u=dim)
        self.H = np.eye(dim)
        self.F = self.f_jacobian

    def f_jacobian(self, x):
        return np.eye(self.dim_x)

    def f(self, x, u):
        self.u_val = u
        return x + self.device_update(x, u)

    def device_update(self, x: np.ndarray, u, use_integral: bool = False):
        """Return difference after program device."""
        if use_integral:
            return self._integral_update(x, u)
        else:
            return self._summation_update(x, u)

    def _integral_update(self, x, u):
        u_up = np.maximum(u, 0)
        u_down = np.minimum(u, 0)
        return (
            integrate.quad(self.step_update, args=True, a=x, b=x + u_up)[0]
            + integrate.quad(self.step_update, args=False, a=x + u_down, b=x)[0]
        )

    def _summation_update(self, x, u):
        u_up = np.maximum(u, 0)
        u_down = np.minimum(u, 0)
        dw_up = self.dw_min * (1 + self.up_down)
        dw_down = self.dw_min * (1 - self.up_down)
        up_iter = int(u_up / dw_up)
        down_iter = int(u_down / dw_down)
        x_up = x_down = x
        for _ in range(up_iter):
            x_up = self.step_update(x_up, up=True)
        for _ in range(down_iter):
            x_down = self.step_update(x_down, up=False)
        return x_up + x_down

    @abstractmethod
    def step_update(self, x: np.ndarray, up: bool):
        pass

    def update(self, z):
        """Update the state estimation of the device.

        Args:
            z (_type_): _description_
        """
        super().update(z, HJacobian=lambda x: np.eye(self.dim_x), Hx=lambda x: x)


class ExpDeviceEKF(BaseDeviceEKF):

    # TODO: Add drift, lifetime(decay rate), clipping
    # TODO: Consider parameter variation
    def __init__(self, dim, **kwargs):
        super().__init_(dim)
        self.F = self.f_jacobian
        # device parameters. See below for details
        # https://aihwkit.readthedocs.io/en/stable/api/aihwkit.simulator.configs.devices.html#aihwkit.simulator.configs.devices.ConstantStepDevice
        self.dw_min = kwargs.get("dw_min")
        self.up_down = kwargs.get("up_down")  # Step size difference between up and down pulses
        self.A_up = kwargs.get("A_up")
        self.A_down = kwargs.get("A_down")
        self.gamma_up = kwargs.get("gamma_up")
        self.gamma_down = kwargs.get("gamma_down")
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.w_min = kwargs.get("w_min")
        self.w_max = kwargs.get("w_max")

    def step_update(self, x: np.ndarray, up: bool):
        """Update the device state for a single pulse."""
        gamma = self.gamma_up if up else self.gamma_down
        A = self.A_up if up else self.A_down
        z = 2 * self.a * x / (self.w_max - self.w_min) + self.b
        y = 1 - A * np.exp(gamma * z)
        return max(0, y)

    def f_jacobian(self, x):
        return np.eye(self.dim_x) + self.step_update(x + self.u_val) - self.step_update(x)


class LinearDeviceEKF(BaseDeviceEKF):

    pass
