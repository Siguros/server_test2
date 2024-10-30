from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed
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
        self.S_scale = None
        self.K_scale = None
        # Solve Riccati equation for S_scale
        # poly = np.array([1, -(self.r), -(self.r) * (self.q)])
        # self.S_scale = np.roots(poly).max()
        # self.K_scale = self.S_scale / (self.S_scale + q)
        self.x_est = None

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

    def get_lqg_gain(self, u):
        return 1


class BaseDeviceEKF(ABC):
    """Extended Kalman filter for device programming.

    Model:

    :math:`x_{k+1} = f(x_{k}, u_k) + w_k`\\
    :math:`z_k = x_k + v_k`
    """

    def __init__(self, dim, read_noise_std: float, update_noise_std: float):
        self.dim = dim
        self.q = read_noise_std**2  # covariance of observation noise
        self.r = update_noise_std**2  # covariance of process noise
        self.P_diag = np.ones(dim)  # diagonal entries of P(Initial covariance matrix)
        self.F_diag = np.ones(dim)
        self.S_diag = None
        self.K_diag = None
        self.x_est = None

    @abstractmethod
    def f_jacobian_x(self, x, u):
        """Jacobian of f at x.

        Returns diagonal entry of jacobian matrix of f at x.
        """
        pass

    @abstractmethod
    def f_jacobian_u(self, x, u):
        """Jacobian of f at u.

        Returns diagonal entry of jacobian matrix of f at u.
        """
        pass

    def f(self, x, u: np.ndarray | None):
        """Transition function."""
        if u is None:
            return x
        else:
            return x + self.device_update(x, u)

    def device_update(self, x: np.ndarray, u, use_integral: bool = True):
        """Return difference after program device."""
        if use_integral:
            return self._integral_update(x, u)
        else:
            return self._summation_update(x, u)

    def _integral_update(self, x, u):
        u_up = np.maximum(u, 0)
        u_down = np.minimum(u, 0)
        raise NotImplementedError
        # return (
        #     integrate.quad(self.step_device_update, a=x, b=x + u_up)[0]
        #     - integrate.quad(self.step_device_downdate, a=x + u_down, b=x)[0]
        # )

    def _summation_update(self, x: np.ndarray, u: np.ndarray):
        u_up = np.maximum(u, 0)
        u_down = np.minimum(u, 0)
        dw_up = self.dw_min * (1 + self.up_down)
        dw_down = self.dw_min * (1 - self.up_down)
        up_iters = (u_up / dw_up).astype(int)
        down_iters = (u_down / dw_down).astype(int)
        x_up = x_down = np.zeros_like(x)

        def update_step(idx, i_it, x, x_up):
            for _ in range(i_it):
                x_up[idx] += self.step_device_update(x[idx] + x_up[idx])
            return x_up[idx]

        def downdate_step(idx, j_it, x, x_down):
            for _ in range(j_it):
                x_down[idx] += self.step_device_downdate(x[idx] - x_down[idx])
            return x_down[idx]

        x_up = Parallel(n_jobs=-1)(
            delayed(update_step)(idx, i_it, x, x_up) for idx, i_it in enumerate(up_iters)
        )
        x_down = Parallel(n_jobs=-1)(
            delayed(downdate_step)(idx, j_it, x, x_down) for idx, j_it in enumerate(down_iters)
        )

        return np.array(x_up) - np.array(x_down)

    @abstractmethod
    def step_device_update(self, x: np.ndarray):
        """Update the device state for a single pulse.

        Returns the difference after the pulse.
        """
        pass

    @abstractmethod
    def step_device_downdate(self, x: np.ndarray):
        """Downdate the device state for a single pulse.

        Returns the absolute difference after the pulse.
        """
        pass

    def update(self, z):
        """Update the state(weight) estimation of the device."""
        y_res = z - self.x_est
        # H_diag = np.ones(self.dim)
        self.S_diag = self.P_diag + self.r
        self.K_diag = self.P_diag / self.S_diag
        self.x_est += self.K_diag * y_res
        self.P_diag = (1 - self.K_diag) * self.P_diag

    def get_lqg_gain(self, u: np.ndarray | None):
        """Return diagonal entries of Linear Quadratic Gaussian(LQG) control gain matrix."""
        if u is None:
            return np.ones(self.dim)
        else:
            B_diag = self.f_jacobian_u(self.x_est, u)
            self.F_diag = self.f_jacobian_x(self.x_est, u)
            return self.F_diag / (B_diag + 1e-8)

    def predict(self, u):
        """Predict the next state(weight) of the device."""
        self.x_est = self.f(self.x_est, u)
        # self.P = F @ self.P @ F.T + self.Q
        self.P_diag = self.P_diag * self.F_diag**2 + self.q


class LinearDeviceEKF(BaseDeviceEKF):

    # TODO: Add drift, lifetime(decay rate), clipping, reverse_down
    # TODO: Consider parameter variation
    def __init__(self, dim, read_noise_std: float, update_noise_std: float, **kwargs):
        super().__init__(dim, read_noise_std, update_noise_std)
        self.update_x_plus_u = None
        # device parameters. See below for details
        # https://aihwkit.readthedocs.io/en/stable/api/aihwkit.simulator.configs.devices.html#aihwkit.simulator.configs.devices.ConstantStepDevice
        self.dw_min = kwargs.get("dw_min")
        self.up_down = kwargs.get("up_down")  # Step size difference between up and down pulses
        self.gamma_up = kwargs.get("gamma_up")
        self.gamma_down = kwargs.get("gamma_down")

        self.w_min = kwargs.get("w_min")
        self.w_max = kwargs.get("w_max")

        self._ghat_up = -abs(self.gamma_up) / self.w_max
        self._ghat_down = -abs(self.gamma_down) / self.w_min

    def _integral_update(self, x, u):
        if self.gamma_down == 0 and self.gamma_up == 0:
            return u
        else:
            u_up = np.maximum(u, 0)
            u_down = np.minimum(u, 0)
            x_up = u_up * (
                1 + (x + 0.5 * u_up) * self._ghat_up
            )  # [x + self._ghat_up*x**2]|_x^{x+u}
            x_down = u_down * (1 + (x + 0.5 * u_down) * self._ghat_down)
            return x_up - x_down

    def step_device_update(self, x: np.ndarray):
        """Update the device state for a single pulse."""
        dw = 1 + self._ghat_up * x
        return dw

    def step_device_downdate(self, x: np.ndarray):
        """Downdate the device state for a single pulse."""
        dw = 1 + self._ghat_down * x
        return dw

    def f_jacobian_u(self, x, u):
        self.update_x_plus_u = (
            self.step_device_update(x + u) + self.step_device_downdate(x + u)
        ) / 2
        return self.update_x_plus_u

    def f_jacobian_x(self, x, u):
        update_x = (self.step_device_update(x) + self.step_device_downdate(x)) / 2
        return np.ones(self.dim) + self.update_x_plus_u - update_x


class ExpDeviceEKF(BaseDeviceEKF):

    # TODO: Add drift, lifetime(decay rate), clipping
    # TODO: Consider parameter variation
    def __init__(self, dim, read_noise_std: float, update_noise_std: float, **kwargs):
        super().__init__(dim, read_noise_std, update_noise_std)
        self.update_x_plus_u = None
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
        self._slope = 2 * self.a / (self.w_max - self.w_min)

    def step_device_update(self, x: np.ndarray):
        """Update the device state for a single pulse."""
        z = self._slope * x + self.b
        y = 1 - self.A_up * np.exp(self.gamma_up * z)
        dw = np.maximum(y, 0)
        return dw

    def step_device_downdate(self, x: np.ndarray):
        """Downdate the device state for a single pulse."""
        z = self._slope * x + self.b
        y = 1 - self.A_down * np.exp(self.gamma_down * z)
        dw = np.maximum(y, 0)
        return dw

    def f_jacobian_u(self, x, u):
        self.update_x_plus_u = (
            self.step_device_update(x + u) + self.step_device_downdate(x + u)
        ) / 2
        return self.update_x_plus_u

    def f_jacobian_x(self, x, u):
        update_x = (self.step_device_update(x) + self.step_device_downdate(x)) / 2
        return np.ones(self.dim) + self.update_x_plus_u - update_x
