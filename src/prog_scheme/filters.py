from abc import ABC, abstractmethod

import numpy as np
import torch
from jaxtyping import Float
from joblib import Parallel, delayed
from scipy import sparse

StateVec = Float[np.ndarray, "out*in"]


class KalmanFilter:
    """Class for Kalman filter."""

    def __init__(self, state_dim: int = 3, obs_dim: int = 2) -> None:
        self.x_est = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        self.F = np.eye(state_dim)
        self.H = np.eye(obs_dim)
        self.Q = np.eye(state_dim)
        self.R = np.eye(obs_dim)
        self.K = np.zeros((state_dim, obs_dim))

    def predict(self, u: StateVec) -> None:
        """Predict the next state of the device using the state transition matrix."""
        self.x_est = self.F @ self.x_est + u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, z: StateVec) -> None:
        """Update the state estimation of the device."""
        y_res = z - self.H @ self.x_est
        S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_est += self.K @ y_res
        self.P = (np.eye(self.x_est.shape[0]) - self.K @ self.H) @ self.P


class ExtendedKalmanFilter(KalmanFilter):
    def __init__(self) -> None:
        super().__init__()
        self.f = None
        self.F = None
        self.H = None

    def predict(self, u: StateVec) -> None:
        """Predict the next state of the device using the state transition matrix."""
        self.x_est = self.f(self.x_est, u)
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, z: StateVec) -> None:
        super().correct(z)

    def f(self, x: StateVec, u: StateVec) -> StateVec:
        """Transition function."""
        pass

    def f_jacobian_x(self, x: StateVec, u: StateVec) -> StateVec:
        """Jacobian of f at x."""
        pass

    def f_jacobian_u(self, x: StateVec, u: StateVec) -> StateVec:
        """Jacobian of f at u."""
        pass


class ErrorStateKalmanFilter(KalmanFilter):
    def __init__(self) -> None:
        super().__init__()
        self.f = None
        self.F = None
        self.H = None
        self.x_err = None
        self.P_err = None

    def predict(self, u: StateVec) -> None:
        """Predict the next state of the device using the state transition matrix."""
        self.x_est = self.f(self.x_est, u)
        self.F = self.f_jacobian_x(self.x_est, u)
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.x_err = np.zeros_like(self.x_est)
        self.P_err = self.F @ self.P_err @ self.F.T + self.Q

    def correct(self, z: StateVec) -> None:
        y_res = z - self.H @ self.x_est
        S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x_err = self.K @ y_res
        self.P_err = (np.eye(self.x_est.shape[0]) - self.K @ self.H) @ self.P
        self.x_est += self.x_err
        self.P = self.P_err

    def f_jacobian_x(self, x: StateVec, u: StateVec) -> StateVec:
        pass

    def f_jacobian_u(self, x: StateVec, u: StateVec) -> StateVec:
        pass


class IteratedErrorStateKalmanFilter(ErrorStateKalmanFilter):
    def __init__(self) -> None:
        super().__init__()
        self.correct_tol: float = 1e-3

    def predict(self, u: StateVec) -> None: ...

    def correct(self, z: StateVec) -> None:
        while True:
            y_res = z - self.H @ self.x_est
            S = self.H @ self.P @ self.H.T + self.R
            self.K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x_err = self.K @ y_res
            self.P_err = (np.eye(self.x_est.shape[0]) - self.K @ self.H) @ self.P
            self.x_est += self.x_err
            self.P = self.P_err
            if np.linalg.norm(self.x_err) < self.correct_tol:
                break


class AbstractDeviceFilter(ABC):
    """Abstract class for device programming with (mainly Kalman) filter."""

    def __init__(self, dim: int, read_noise_std: float, update_noise_std: float, **kwargs):
        self.dim = dim
        self.x_est = np.empty(dim)
        self.q = update_noise_std**2  # covariance of process noise
        self.r = read_noise_std**2  # covariance of measurement noise

    @abstractmethod
    def predict(self, u: StateVec) -> None:
        """Predict the next state of the device."""
        ...

    @abstractmethod
    def correct(self, z: StateVec) -> None:
        """Update the state estimation of the device after reading the device state."""
        ...

    def get_x_est(self) -> torch.Tensor:
        """Return the copied tensor of current state estimation of the device."""
        return torch.tensor(self.x_est)


class NoFilter(AbstractDeviceFilter):
    """Class for device programming without any filtering."""

    def __init__(self, dim: int = 0, read_noise_std: float = 0.0, update_noise_std: float = 0.0):
        self.dim = dim
        self.x_est = None

    def predict(self, u: StateVec) -> None:
        """Predict the next state of the device without any filtering."""
        self.x_est += u

    def correct(self, z: StateVec) -> None:
        """Update the state estimation of the device without any filtering."""
        self.x_est = z


class DeviceKF(AbstractDeviceFilter):
    """Class for device programming with Kalman filter.

    x_k = x_{k-1} + u_k + w_k z_k = x_k + v_k
    """

    def __init__(self, dim: int, read_noise_std: float, update_noise_std: float):
        super().__init__(dim, read_noise_std, update_noise_std)
        self.P_scale = 1
        # Solve Riccati equation for S_scale
        # poly = np.array([1, -(self.r), -(self.r) * (self.q)])
        # self.S_scale = np.roots(poly).max()
        # self.K_scale = self.S_scale / (self.S_scale + q)

    def predict(self, u: StateVec):
        """Predict the next state of the device."""
        self.x_est += u
        self.P_scale += self.q

    def correct(self, z: StateVec):
        """Update the state estimation of the device after reading the device state."""
        y_res = z - self.x_est
        S_scale = self.P_scale + self.r
        K_scale = self.P_scale / S_scale
        self.x_est += K_scale * y_res
        self.P_scale = (1 - K_scale) * self.P_scale


class DeviceProjKF(AbstractDeviceFilter):
    """Class for device programming with Kalman filter with projected update."""

    def __init__(
        self,
        dim: int,
        read_noise_std: float,
        update_noise_std: float,
        batch_size: int,
        output_dim: int,
    ):
        super().__init__(dim, read_noise_std, update_noise_std)
        self.P = np.eye(output_dim)

    def predict(self, u: StateVec):
        """Predict the next state of the device."""
        self.x_est += u
        self.P += self.q

    def correct(
        self,
        z_proj: Float[np.ndarray, "batch out"],  # noqa: F722
        x_input: Float[np.ndarray, "batch in"],  # noqa: F722
    ) -> None:
        """Update the state estimation of the device from the projected(partial) measurement.

        Assume observation dynamics as below:
        z_proj = x_inp@Z = x_inp@(W + v)
        after vectorization,
        \vec(z_proj) = P@X_hat + v,

        where W is the true state, v is the read noise, and self.x_est=\vec{Z}
        In case of the programming weights, z_proj = tile.forward(x_inp).
        """
        output_dim = z_proj.shape[1]
        input_dim = x_input.shape[1]
        Proj = x_input
        x_est_mtx = self.x_est.reshape(input_dim, output_dim)
        # Measurement residual (b x o matrix)
        residual = z_proj - Proj @ x_est_mtx
        # Innovation covariance S (b x b matrix)
        S = self.P * Proj @ Proj.T + self.r
        # Kalman gain K (o x b matrix)
        K_k = self.P @ Proj.T @ np.linalg.inv(S)
        # Update state estimate X_hat_new (o x o matrix)
        x_est_mtx += K_k @ residual
        self.x_est = x_est_mtx.flatten()
        # Update state covariance P_xx_new (o x o matrix)
        self.P -= K_k @ S @ K_k.T


class BaseDeviceEKF(AbstractDeviceFilter):
    """Extended Kalman filter for pusled device programming.

    Model:

    :math:`x_{k+1} = f(x_{k}, u_k) + w_k`\\
    :math:`z_k = x_k + v_k`
    """

    def __init__(self, dim, read_noise_std: float, update_noise_std: float, iterative_update: bool):
        super().__init__(dim, read_noise_std, update_noise_std)
        self.iterative_update = iterative_update
        # diagonal entries of P(Initial covariance matrix)
        self.P_diag = np.ones(dim)
        self.F_diag = np.ones(dim)
        self.K_diag = None
        self.x_est = None

        self._f_out = None

    @abstractmethod
    def f_jacobian_x(self, x: StateVec, u: StateVec) -> StateVec:
        """Jacobian of f at x.

        Returns diagonal entry of jacobian matrix of f at x.
        """
        pass

    @abstractmethod
    def f_jacobian_u(self, x: StateVec, u: StateVec) -> StateVec:
        """Jacobian of f at u.

        Returns diagonal entry of jacobian matrix of f at u.
        """
        pass

    def f(
        self,
        x: StateVec,
        u: StateVec | None,
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

    def _iterative_update(self, x: StateVec, u: StateVec):
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

    def _summation_update(self, x: StateVec, u: StateVec):
        """Updates device state at once."""
        raise NotImplementedError

    @abstractmethod
    def device_update_once(self, x: StateVec):
        """Update the device state for a single pulse.

        Returns the difference after the pulse.
        """
        pass

    @abstractmethod
    def device_downdate_once(self, x: StateVec):
        """Downdate the device state for a single pulse.

        Returns the absolute difference after the pulse.
        """
        pass

    def get_lqg_gain(self, u: StateVec | None):
        """Return Linear Quadratic Gaussian (LQG) control gain using Direct Collocation."""

        # if u is None:
        #     return np.ones(self.dim)
        # else:
        #     # Cost function weights
        #     # Q = np.eye(self.dim) # State cost
        #     # R = np.zeros(len(u))

        #     # Initial guess for control inputs
        #     U0 = np.tile(u, (N, 1)).flatten()

        #     # Define the cost function
        #     def cost(U_flat):
        #         U = U_flat.reshape(N, -1)
        #         xk = x0
        #         total_cost = xk**2
        #         for k in range(N):
        #             # State update using Euler integration
        #             uk = U[k]
        #             xk1 = xk + dt * self.f(xk, uk)
        #             xk = xk1
        #             # Accumulate cost
        #             total_cost += xk**2
        #         return total_cost

        #     # Optimize control inputs
        #     res = minimize(cost, U0, method="SLSQP")

        #     # Optimal control at the first timestep
        #     U_opt = res.x.reshape(N, -1)
        #     u0_opt = U_opt[0]

        #     # Estimate gain K assuming linear control law u = -K * x
        #     K = np.linalg.pinv(x0) @ u0_opt

        #     return K

    def _solve_dre(
        self,
        S: StateVec,
        x: StateVec,
        u: StateVec,
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

    def predict(self, u: StateVec):
        """Predict the next state(weight) of the device."""
        self.F_diag = self.f_jacobian_x(self.x_est, u)
        self.x_est = self.f(self.x_est, u, store_f_out=True)
        # self.P = F @ self.P @ F.T + self.Q
        self.P_diag = self.P_diag * self.F_diag**2 + self.q

    def correct(self, z: StateVec):
        """Update the state(weight) estimation of the device."""
        y_res = z - self.x_est
        # H_diag = np.ones(self.dim)
        self.K_diag = self.P_diag / (self.P_diag + self.r)
        self.x_est += self.K_diag * y_res
        self.P_diag = (1 - self.K_diag) * self.P_diag


class LinearDeviceEKF(BaseDeviceEKF):
    # TODO: Add drift, lifetime(decay rate), clipping, reverse_down
    # TODO: Consider parameter variation
    def __init__(
        self, dim, read_noise_std: float, update_noise_std: float, iterative_update: bool, **kwargs
    ):
        """Extended Kalman filter for linear device programming.

        Args:
        - dim: int, dimension of the device state
        - read_noise_std: float, standard deviation of the read noise
        - update_noise_std: float, standard deviation of the update noise
        - iterative_update: bool, whether to iteratively update or update at once
        - kwargs: dict, device parameters
        """
        super().__init__(dim, read_noise_std, update_noise_std, iterative_update)
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

    def _summation_update(self, x: StateVec, u: StateVec):
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

    def device_update_once(self, x: StateVec):
        """Update the device state for a single pulse."""
        dw = self._slope_up * x + self._scale_up
        return dw

    def device_downdate_once(self, x: StateVec):
        """Downdate the device state for a single pulse."""
        dw = self._slope_down * x + self._scale_down

        return dw

    def f_jacobian_u(self, x: StateVec, u: StateVec):
        # f(x,u) = (x - b/(1-r)) * r^k + b/(1-r), k = n_up & n_down, r = 1 + slope_up, b = scale_up
        # df/du = (x - b/(1-r)) * r^k * ln(r) / b
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
        return df_du

    def f_jacobian_x(self, x: StateVec, u: StateVec):
        # df/dx = r^k, where r = 1 + slope_up, k = n_up
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

        return df_dx


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

    def device_update_once(self, x: StateVec):
        """Update the device state for a single pulse."""
        z = self._slope * x + self.b
        y = 1 - self.A_up * np.exp(self.gamma_up * z)
        dw = np.maximum(y, 0)
        return dw

    def device_downdate_once(self, x: StateVec):
        """Downdate the device state for a single pulse."""
        z = self._slope * x + self.b
        y = 1 - self.A_down * np.exp(self.gamma_down * z)
        dw = np.maximum(y, 0)
        return dw

    def f_jacobian_u(self, x: StateVec, u: StateVec):
        self.update_x_plus_u = (
            self.device_update_once(self._f_out) + self.device_downdate_once(self._f_out)
        ) / 2
        return self.update_x_plus_u

    def f_jacobian_x(self, x: StateVec, u: StateVec):
        update_x = (self.device_update_once(x) + self.device_downdate_once(x)) / 2
        return np.ones(self.dim) + self.update_x_plus_u - update_x
