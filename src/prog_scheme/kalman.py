from abc import ABC, abstractmethod

import numpy as np
from jaxtyping import Float
from joblib import Parallel, delayed
from scipy import sparse
from scipy.optimize import minimize


class AbstractDeviceFilternCtrl(ABC):
    """Abstract class for device programming with Kalman filter and control."""

    @abstractmethod
    def __init__(self, dim: int, read_noise_std: float, update_noise_std: float, **kwargs): ...

    @abstractmethod
    def predict(self, u: Float[np.ndarray, "dim1"]) -> None:
        """Predict the next state of the device."""
        ...

    @abstractmethod
    def update(self, z: Float[np.ndarray, "dim1"]) -> None:
        """Update the state estimation of the device after reading the device state."""
        ...

    @abstractmethod
    def get_lqg_gain(self, u) -> Float[np.ndarray, "dim1"] | int:
        """Return diagonal entries of Linear Quadratic Gaussian(LQG) control gain matrix."""
        ...


class NoFilter(AbstractDeviceFilternCtrl):
    """Class for device programming without any filtering."""

    def __init__(self, dim: int, read_noise_std: float = 0.0, update_noise_std: float = 0.0):
        self.dim = dim
        self.x_est = None

    def predict(self, u: Float[np.ndarray, "dim1"]) -> None:
        """Predict the next state of the device without any filtering."""
        self.x_est += u

    def update(self, z: Float[np.ndarray, "dim1"]) -> None:
        """Update the state estimation of the device without any filtering."""
        self.x_est = z

    def get_lqg_gain(self, u) -> Float[np.ndarray, "dim1"] | int:
        """Return a default gain of 1."""
        return 1


class DeviceKF(AbstractDeviceFilternCtrl):
    """Class for device programming with Kalman filter.

    x_k = x_{k-1} + u_k + w_k z_k = x_k + v_k
    """

    def __init__(self, dim: int, read_noise_std: float, update_noise_std: float):
        self.dim = dim
        self.q = update_noise_std**2
        self.r = read_noise_std**2
        self.P_scale = 1
        self.S_scale = None
        self.K_scale = None
        # Solve Riccati equation for S_scale
        # poly = np.array([1, -(self.r), -(self.r) * (self.q)])
        # self.S_scale = np.roots(poly).max()
        # self.K_scale = self.S_scale / (self.S_scale + q)
        self.x_est = None

    def predict(self, u: Float[np.ndarray, "dim1"]):
        """Predict the next state of the device."""
        self.x_est += u
        self.P_scale += self.q

    def update(self, z: Float[np.ndarray, "dim1"]):
        """Update the state estimation of the device after reading the device state."""
        y_res = z - self.x_est
        self.S_scale = self.P_scale + self.r
        self.K_scale = self.P_scale / self.S_scale
        self.x_est += self.K_scale * y_res
        self.P_scale = (1 - self.K_scale) * self.P_scale

    def get_lqg_gain(self, u):
        return 1


class BaseDeviceEKF(AbstractDeviceFilternCtrl):
    """Extended Kalman filter for pusled device programming.

    Model:

    :math:`x_{k+1} = f(x_{k}, u_k) + w_k`\\
    :math:`z_k = x_k + v_k`
    """

    def __init__(
        self, dim, read_noise_std: float, update_noise_std: float, iterative_update: bool
    ):
        self.dim = dim
        self.q = update_noise_std**2  # covariance of process noise
        self.r = read_noise_std**2  # covariance of measurement noise
        self.iterative_update = iterative_update
        # diagonal entries of P(Initial covariance matrix)
        self.P_diag = np.ones(dim)
        self.F_diag = np.ones(dim)
        self.S_diag = None
        self.K_diag = None
        self.x_est = None

        self._f_out = None

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

    def f(
        self,
        x: Float[np.ndarray, "dim1"],
        u: Float[np.ndarray, "dim1"] | None,
        store_f_out: bool = False,
    ):
        """Transition function.

        Optionally store the output of the function for jacobian calculation until the next call.
        """
        if u is None:
            out = x
        else:
            if self.iterative_update:
                out = self._iterative_update(x, u)
            else:
                out = self._summation_update(x, u)
        assert not np.all(
            out == self._f_out
        ), "f(x,u) is not updated. Maybe you called f(x,u) before."
        self._f_out = out if store_f_out else None
        return out

    def _iterative_update(self, x: Float[np.ndarray, "dim1"], u: Float[np.ndarray, "dim1"]):
        """Iteratively updates device state."""
        u_up = np.maximum(u, 0)
        u_down = np.minimum(u, 0)
        dw_up = self.dw_min * (1 + self.up_down)
        dw_down = self.dw_min * (1 - self.up_down)
        up_iters = (u_up / dw_up).astype(int)
        down_iters = (u_down / dw_down).astype(int)
        x_up = x_down = np.zeros_like(x)

        def update_step(idx, i_it, x, x_up):
            for _ in range(i_it):
                x_up[idx] += self.device_update_once(x[idx] + x_up[idx])
            return x_up[idx]

        def downdate_step(idx, j_it, x, x_down):
            for _ in range(j_it):
                x_down[idx] += self.device_downdate_once(x[idx] - x_down[idx])
            return x_down[idx]

        x_up = Parallel(n_jobs=-1)(
            delayed(update_step)(idx, i_it, x, x_up) for idx, i_it in enumerate(up_iters)
        )
        x_down = Parallel(n_jobs=-1)(
            delayed(downdate_step)(idx, j_it, x, x_down) for idx, j_it in enumerate(down_iters)
        )

        return np.array(x_up) + np.array(x_down) + x

    def _summation_update(self, x: Float[np.ndarray, "dim1"], u: Float[np.ndarray, "dim1"]):
        """Updates device state at once."""
        raise NotImplementedError

    @abstractmethod
    def device_update_once(self, x: Float[np.ndarray, "dim1"]):
        """Update the device state for a single pulse.

        Returns the difference after the pulse.
        """
        pass

    @abstractmethod
    def device_downdate_once(self, x: Float[np.ndarray, "dim1"]):
        """Downdate the device state for a single pulse.

        Returns the absolute difference after the pulse.
        """
        pass

    def get_lqg_gain(self, u: Float[np.ndarray, "dim1"] | None):
        """Return Linear Quadratic Gaussian (LQG) control gain using Direct Collocation."""
        if u is None:
            return np.ones(self.dim)
        else:
            # Cost function weights
            # Q = np.eye(self.dim) # State cost
            # R = np.zeros(len(u))

            # Initial guess for control inputs
            U0 = np.tile(u, (N, 1)).flatten()

            # Define the cost function
            def cost(U_flat):
                U = U_flat.reshape(N, -1)
                xk = x0
                total_cost = xk**2
                for k in range(N):
                    # State update using Euler integration
                    uk = U[k]
                    xk1 = xk + dt * self.f(xk, uk)
                    xk = xk1
                    # Accumulate cost
                    total_cost += xk**2
                return total_cost

            # Optimize control inputs
            res = minimize(cost, U0, method="SLSQP")

            # Optimal control at the first timestep
            U_opt = res.x.reshape(N, -1)
            u0_opt = U_opt[0]

            # Estimate gain K assuming linear control law u = -K * x
            K = np.linalg.pinv(x0) @ u0_opt

            return K

    def _solve_dre(
        self,
        S: Float[np.ndarray, "dim1"],
        x: Float[np.ndarray, "dim1"],
        u: Float[np.ndarray, "dim1"],
        Q_perf: float,
        R_perf: float,
    ):
        """Solve Discrete Riccati Equation for LQG control gain.

        Modified from scipy.linalg.solve_discrete_are

        The DARE is defined as

        .. math::

          S' = A^H@S@A - (A^H@S@B) (R + B^H@S@B)^{-1} (B^H@S@A) + Q
        """
        A = sparse.diags_array(self.f_jacobian_x(x, u))
        B = sparse.diags_array(self.f_jacobian_u(x, u))
        # res = _are_validate_args(A, B, Q_perf, R_perf)
        return (
            A.T @ S @ A
            - (A.T @ S @ B) @ np.linalg.inv(R_perf + B.T @ S @ B) @ (B.T @ S @ A)
            + Q_perf
        )

    def predict(self, u: Float[np.ndarray, "dim1"]):
        """Predict the next state(weight) of the device."""
        self.x_est = self.f(self.x_est, u, store_f_out=True)
        # self.P = F @ self.P @ F.T + self.Q
        self.P_diag = self.P_diag * self.F_diag**2 + self.q

    def update(self, z: Float[np.ndarray, "dim1"]):
        """Update the state(weight) estimation of the device."""
        y_res = z - self.x_est
        # H_diag = np.ones(self.dim)
        self.S_diag = self.P_diag + self.r
        self.K_diag = self.P_diag / self.S_diag
        self.x_est += self.K_diag * y_res
        self.P_diag = (1 - self.K_diag) * self.P_diag


class LinearDeviceEKF(BaseDeviceEKF):

    # TODO: Add drift, lifetime(decay rate), clipping, reverse_down
    # TODO: Consider parameter variation
    def __init__(
        self, dim, read_noise_std: float, update_noise_std: float, iterative_update: bool, **kwargs
    ):
        super().__init__(dim, read_noise_std, update_noise_std, iterative_update)
        self.update_x_plus_u = None
        # device parameters. See below for details
        # https://aihwkit.readthedocs.io/en/stable/api/aihwkit.simulator.configs.devices.html#aihwkit.simulator.configs.devices.ConstantStepDevice
        self.dw_min = kwargs.get("dw_min")
        # Step size difference between up and down pulses
        self.up_down = kwargs.get("up_down")
        self.gamma_up = kwargs.get("gamma_up")
        self.gamma_down = kwargs.get("gamma_down")

        self.w_min = kwargs.get("w_min")
        self.w_max = kwargs.get("w_max")

        self._scale_up = (self.up_down + 1) * self.dw_min  # step size for up pulse
        self._scale_down = (-self.up_down + 1) * self.dw_min
        self._slope_up = -abs(self.gamma_up) * self._scale_up / self.w_max
        self._slope_down = -abs(self.gamma_down) * self._scale_down / self.w_min

    def _summation_update(self, x: Float[np.ndarray, "dim1"], u: Float[np.ndarray, "dim1"]):
        if self.gamma_down == 0 and self.gamma_up == 0:
            return u
        else:
            # w_k+1 = r*w_k + b, where r = 1 + slope_up, b = scale_up
            # w_k = (w_0 - b/(1-r)) * r^k + b/(1-r)

            # u_up 계산 (u ≥ 0)
            u_up = np.maximum(u, 0)
            r_up = 1 - self._slope_down  # slope_down 사용
            b_up = self._scale_down  # scale_down 사용
            n_up = np.floor(u_up / b_up).astype(int)  # 정수 부분 계산
            x_up = (x - b_up / (1 - r_up)) * r_up**n_up + b_up / (1 - r_up)

            # u_down 계산 (u ≤ 0)
            u_down = np.minimum(u, 0)
            r_down = 1 + self._slope_up  # slope_up 사용
            b_down = self._scale_up  # scale_up 사용
            n_down = np.floor(u_down / b_down).astype(int)
            x_down = (x - b_down / (1 - r_down)) * r_down**n_down + b_down / (1 - r_down)
            result = x_up + x_down - x

        return result

    def device_update_once(self, x: Float[np.ndarray, "dim1"]):
        """Update the device state for a single pulse."""
        dw = self._slope_up * x + self._scale_up
        return dw

    def device_downdate_once(self, x: Float[np.ndarray, "dim1"]):
        """Downdate the device state for a single pulse."""
        dw = self._slope_down * x + self._scale_down

        return dw

    def compute_jacobians(
        self, x: Float[np.ndarray, "dim1"], u: Float[np.ndarray, "dim1"]
    ) -> tuple[np.ndarray, Float[np.ndarray, "dim1"]]:
        """Computes the Jacobian matrices df/dx and df/du of the _summation_update function with
        respect to x and u.

        Parameters:
        - x: Float[np.ndarray,"dim1"], state vector
        - u: Float[np.ndarray,"dim1"], control input vector

        Returns:
        - df_dx_matrix: Float[np.ndarray,"dim1"], Jacobian matrix of shape (n, n) with respect to x
        - df_du_matrix: Float[np.ndarray,"dim1"], Jacobian matrix of shape (n, n) with respect to u
        """
        # f(x, u) = _summation_update(x, u)
        # f(x,u) = (x - b/(1-r)) * r^k + b/(1-r), k = n_up & n_down, r = 1 + slope_up, b = scale_up
        # df/dx = r^k, df/du = A * r^k * ln(r) / b, A = x - b/(1-r)
        # u > 0, u <0 case로 나누어 계산
        # Number of states/inputs
        n = x.size

        # Constants
        r_up = 1 - self._slope_down  # Using self._slope_down
        b_up = self._scale_down  # Using self._scale_down
        ln_r_up = np.log(r_up)

        r_down = 1 + self._slope_up  # Using self._slope_up
        b_down = self._scale_up  # Using self._scale_up
        ln_r_down = np.log(r_down)

        # Precompute constants
        denom_up = 1 - r_up
        denom_down = 1 - r_down
        A_up = x - b_up / denom_up
        A_down = x - b_down / denom_down

        # Compute u_up and u_down
        u_up = np.maximum(u, 0)
        u_down = np.minimum(u, 0)

        # Approximate n_up and n_down
        n_up = np.floor(u_up / b_up).astype(int)
        n_down = np.floor(u_down / b_down).astype(int)

        # Compute r_up ** n_up and r_down ** n_down
        r_up_pow_n_up = np.power(r_up, n_up)
        r_down_pow_n_down = np.power(r_down, n_down)

        # Initialize df_dx_vector and df_du_vector
        df_dx_vector = np.zeros(n)
        df_du_vector = np.zeros(n)

        # Indices for u ≥ 0 and u ≤ 0
        indices_up = u >= 0
        indices_down = u <= 0

        # Compute df/dx and df/du for u ≥ 0
        if np.any(indices_up):
            # df/dx
            df_dx_vector[indices_up] = r_up_pow_n_up[indices_up]

            # df/du
            df_du_vector[indices_up] = (
                A_up[indices_up] * r_up_pow_n_up[indices_up] * ln_r_up / b_up
            )

        # Compute df/dx and df/du for u ≤ 0
        if np.any(indices_down):
            # df/dx
            df_dx_vector[indices_down] += r_down_pow_n_down[indices_down]

            # df/du
            df_du_vector[indices_down] = (
                A_down[indices_down] * r_down_pow_n_down[indices_down] * ln_r_down / b_down
            )

        # Subtract 1 from df_dx_vector (from df/dx = derivative of x_up + x_down - x)
        df_dx_vector -= 1

        # Construct the Jacobian matrices as diagonal matrices
        df_dx_matrix = np.diag(df_dx_vector)
        df_du_matrix = np.diag(df_du_vector)

        return df_dx_matrix, df_du_matrix

    def f_jacobian_u(self, x: Float[np.ndarray, "dim1"], u: Float[np.ndarray, "dim1"]):
        # f(x, u) = x + _summation_update(x, u)
        # f(x,u) = (x - b/(1-r)) * r^k + b/(1-r), k = n_up & n_down, r = 1 + slope_up, b = scale_up
        # df/du = A * r^k * ln(r) / b, A = x - b/(1-r)
        # Initialize df_du
        df_du = np.zeros_like(u)

        # Constants
        r_up = 1 - self._slope_down  # Using self._slope_down
        b_up = self._scale_down  # Using self._scale_down
        ln_r_up = np.log(r_up)

        r_down = 1 + self._slope_up  # Using self._slope_up
        b_down = self._scale_up  # Using self._scale_up
        ln_r_down = np.log(r_down)

        A_up = x - b_up / (1 - r_up)
        A_down = x - b_down / (1 - r_down)

        # Compute u_up and u_down
        u_up = np.maximum(u, 0)
        u_down = np.minimum(u, 0)

        # Approximate n_up and n_down
        n_up = np.floor(u_up / b_up).astype(int)
        n_down = np.floor(u_down / b_down).astype(int)

        # Compute r_up ** n_up and r_down ** n_down
        r_up_pow_n_up = np.power(r_up, n_up)
        r_down_pow_n_down = np.power(r_down, n_down)

        # Initialize df_du
        df_du = np.zeros_like(u)

        # Compute df/du for u ≥ 0
        indices_up = u >= 0
        if np.any(indices_up):
            df_du[indices_up] = A_up[indices_up] * r_up_pow_n_up[indices_up] * ln_r_up / b_up

        # Compute df/du for u ≤ 0
        indices_down = u <= 0
        if np.any(indices_down):
            df_du[indices_down] = (
                A_down[indices_down] * r_down_pow_n_down[indices_down] * ln_r_down / b_down
            )

        # The final derivative df_du combines both cases

        df_du_matrix = np.diag(df_du)
        return df_du_matrix

    def f_jacobian_x(self, x: Float[np.ndarray, "dim1"], u: Float[np.ndarray, "dim1"]):
        # Constants
        r_up = 1 - self._slope_down  # Using self._slope_down
        b_up = self._scale_down  # Using self._scale_down

        r_down = 1 + self._slope_up  # Using self._slope_up
        b_down = self._scale_up  # Using self._scale_up

        # Compute u_up and u_down
        u_up = np.maximum(u, 0)
        u_down = np.minimum(u, 0)

        # Approximate n_up and n_down
        n_up = u_up / b_up
        n_down = u_down / b_down

        # Compute r_up ** n_up and r_down ** n_down
        r_up_pow_n_up = np.power(r_up, n_up)
        r_down_pow_n_down = np.power(r_down, n_down)

        # Initialize df_dx
        df_dx = np.zeros_like(u)

        # Compute df/dx for u ≥ 0
        indices_up = u >= 0
        if np.any(indices_up):
            df_dx[indices_up] = r_up_pow_n_up[indices_up]

        # Compute df/dx for u ≤ 0
        indices_down = u <= 0
        if np.any(indices_down):
            df_dx[indices_down] = r_down_pow_n_down[indices_down]

        df_dx_matrix = np.diag(df_dx)
        return df_dx_matrix


class ExpDeviceEKF(BaseDeviceEKF):

    # TODO: Add drift, lifetime(decay rate), clipping
    # TODO: Consider parameter variation
    def __init__(
        self, dim, read_noise_std: float, update_noise_std: float, iterative_update: bool, **kwargs
    ):
        super().__init__(dim, read_noise_std, update_noise_std, iterative_update)
        self.update_x_plus_u = None
        # device parameters. See below for details
        # https://aihwkit.readthedocs.io/en/stable/api/aihwkit.simulator.configs.devices.html#aihwkit.simulator.configs.devices.ConstantStepDevice
        self.dw_min = kwargs.get("dw_min")
        # Step size difference between up and down pulses
        self.up_down = kwargs.get("up_down")
        self.A_up = kwargs.get("A_up")
        self.A_down = kwargs.get("A_down")
        self.gamma_up = kwargs.get("gamma_up")
        self.gamma_down = kwargs.get("gamma_down")
        self.a = kwargs.get("a")
        self.b = kwargs.get("b")
        self.w_min = kwargs.get("w_min")
        self.w_max = kwargs.get("w_max")
        self._slope = 2 * self.a / (self.w_max - self.w_min)

    def device_update_once(self, x: Float[np.ndarray, "dim1"]):
        """Update the device state for a single pulse."""
        z = self._slope * x + self.b
        y = 1 - self.A_up * np.exp(self.gamma_up * z)
        dw = np.maximum(y, 0)
        return dw

    def device_downdate_once(self, x: Float[np.ndarray, "dim1"]):
        """Downdate the device state for a single pulse."""
        z = self._slope * x + self.b
        y = 1 - self.A_down * np.exp(self.gamma_down * z)
        dw = np.maximum(y, 0)
        return dw

    def f_jacobian_u(self, x: Float[np.ndarray, "dim1"], u: Float[np.ndarray, "dim1"]):
        self.update_x_plus_u = (
            self.device_update_once(self._f_out) + self.device_downdate_once(self._f_out)
        ) / 2
        return self.update_x_plus_u

    def f_jacobian_x(self, x: Float[np.ndarray, "dim1"], u: Float[np.ndarray, "dim1"]):
        update_x = (self.device_update_once(x) + self.device_downdate_once(x)) / 2
        return np.ones(self.dim) + self.update_x_plus_u - update_x
