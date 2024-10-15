from typing import Any, Optional, Union

import numpy as np
import torch
from filterpy.kalman import ExtendedKalmanFilter
from scipy import linalg as sla
from torch import Tensor

from src.utils.pylogger import RankedLogger

log = RankedLogger(rank_zero_only=True)


@torch.no_grad()
def gdp2(
    self,
    batch_size: int = 5,
    learning_rate: float = 1,
    max_iter: int = 100,
    tolerance: Optional[float] = 0.01,
    w_init: Union[float, Tensor] = 0.0,
    norm_type: str = "nuc",
    **kwargs: Any,
) -> None:
    """Program the target weights into the conductances using the pulse update defined.

    Variable batch version of the `program_weights_gdp` method.
    """

    self.actual_weight_updates = []
    self.desired_weight_updates = []
    target_weights = self.tile.get_weights()

    input_size = self.tile.get_x_size()
    x_values = torch.eye(input_size)
    x_values = x_values.to(self.device)
    target_values = x_values @ target_weights.to(self.device).T

    target_max = target_values.abs().max().item()
    if isinstance(w_init, Tensor):
        self.tile.set_weights(w_init)
    else:
        self.tile.set_weights_uniform_random(-w_init, w_init)  # type: ignore

    lr_save = self.tile.get_learning_rate()  # type: ignore
    self.tile.set_learning_rate(learning_rate)  # type: ignore
    self.initial_weights = self.tile.get_weights().clone()

    for i in range(max_iter):

        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        current_weight = self.tile.get_weights().clone()

        if end_idx > len(x_values):
            # Calculate how much we exceed the length
            exceed_length = end_idx - len(x_values)

            # Slice the arrays and concatenate the exceeded part from the beginning
            x = torch.concatenate((x_values[start_idx:], x_values[:exceed_length]))
            target = torch.concatenate((target_values[start_idx:], target_values[:exceed_length]))
        else:
            x = x_values[start_idx:end_idx]
            target = target_values[start_idx:end_idx]

        y = self.tile.forward(x, False)
        error = y - target
        err_normalized = error.abs().mean().item() / target_max
        mtx_diff = self.tile.get_weights() - target_weights
        norm = torch.linalg.matrix_norm(mtx_diff, ord=norm_type)
        log.debug(f"Error: {norm}")
        # log.debug(f"Error: {err_normalized}")
        """
        if tolerance is not None and norm < tolerance:
            break
        """
        self.tile.update(x, error, False)  # type: ignore
        updated_weight = self.tile.get_weights().clone()
        self.actual_weight_updates.append(updated_weight - current_weight)
        self.desired_weight_updates.append(-error.T @ x)
    self.tile.set_learning_rate(lr_save)  # type: ignore


def compensate_half_selection(v: Tensor) -> Tensor:
    """Compensate the half-selection effect for a vector.

    Args:
        v: Vector to compensate.

    Returns:
        Compensated vector.
    """
    return v


@torch.no_grad()
def svd(
    self,
    max_iter: int = 100,
    use_rank_as_criterion: bool = False,
    tolerance: Optional[float] = 0.01,
    w_init: Union[float, Tensor] = 0.0,
    rank_atol: Optional[float] = 1e-2,
    svd_every_k_iter: int = 1,
    norm_type: str = "nuc",
    **kwargs: Any,
) -> None:
    """Perform singular value decomposition (SVD) based weight programming.

    Args:
        use_rank_as_criterion (bool, optional): Use rank as stopping criterion. If False, use max_iter. Defaults to False.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tolerance (float, optional): Tolerance for convergence. Defaults to 0.01.
        w_init (Union[float, Tensor], optional): Initial value for weights. Defaults to 0.01.
        rank_atol (float, optional): Absolute tolerance for numerical rank computation. Defaults to 1e-6.
        svd_every_k_iter (int, optional): indicating whether to perform SVD every k iterations. Defaults to 1.
        norm_type (str, optional): Type of matrix norm to use. Defaults to "nuc".
        **kwargs: Additional keyword arguments.
    Returns:
        None
    """
    target_weights = self.tile.get_weights()
    self.actual_weight_updates = []
    self.desired_weight_updates = []
    # target_weights = self.tile.get_weights() if target_weights is None else target_weights

    if isinstance(w_init, Tensor):
        self.tile.set_weights(w_init)
    else:
        self.tile.set_weights_uniform_random(-w_init, w_init)  # type: ignore

    self.initial_weights = self.tile.get_weights().clone()

    lr_save = self.tile.get_learning_rate()  # type: ignore
    # x_values = torch.eye(self.tile.get_x_size())
    # x_values = x_values.to(self.device)
    # target_values = x_values @ target_weights.to(self.device).T
    # target_max = target_values.abs().max().item()
    self.tile.set_learning_rate(1)  # type: ignore
    # since tile.update() updates w -= lr*delta_w so flip the sign
    diff = self.read_weights()[0] - target_weights
    U, S, Vh = torch.linalg.svd(diff.double(), full_matrices=False)
    rank = torch.linalg.matrix_rank(diff)
    # if rank_atol is None:
    #     rank_atol = S.max() * max(diff.shape) * torch.finfo(S.dtype).eps
    # rank = torch.sum(S > rank_atol).item()
    max_iter = min(max_iter, rank) if use_rank_as_criterion else max_iter
    for iter in range(max_iter):
        current_weight = self.tile.get_weights().clone()
        i = iter % svd_every_k_iter
        u = U[:, i]
        v = Vh[i, :]
        # uv_ratio = torch.sqrt(u/v)
        uv_ratio = 1
        sqrt_s = torch.sqrt(S[i])
        v *= uv_ratio * sqrt_s
        u *= sqrt_s / uv_ratio
        u1, v1 = compensate_half_selection(u), compensate_half_selection(v)
        self.tile.update(v1.float(), u1.float(), False)
        updated_weight = self.tile.get_weights().clone()

        # realistic weight readout
        diff = self.read_weights()[0] - target_weights
        norm = torch.linalg.matrix_norm(diff, ord=norm_type)
        log.debug(f"Error: {norm}")
        self.actual_weight_updates.append(updated_weight - current_weight)
        self.desired_weight_updates.append(-torch.outer(u1, v1))
        # y = self.tile.forward(x_values, False)
        # # TODO: error와 weight 2norm 사이 관계 분석
        # error = y - target_values
        # err_normalized = error.abs().mean().item() / target_max
        # log.debug(f"Error: {err_normalized}")

        if tolerance is not None and norm < tolerance:
            break
        else:
            pass
        if (iter + 1) % svd_every_k_iter == 0:
            U, S, Vh = torch.linalg.svd(diff.double(), full_matrices=False)

    self.tile.set_learning_rate(lr_save)  # type: ignore


@torch.no_grad()
def lqg_svd(
    self,
    max_iter: int = 100,
    tolerance: Optional[float] = 0.01,
    w_init: Union[float, Tensor] = 0.0,
    norm_type: str = "nuc",
    read_noise_std: float = 0.1,
    update_noise_std: float = 0.1,
    **kwargs: Any,
) -> None:
    """Perform LQG-based weight programming using SVD.

    Args:
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tolerance (Optional[float], optional): Tolerance for convergence. Defaults to 0.01.
        w_init (Union[float, Tensor], optional): Initial value for weights. Defaults to 0.01.
        norm_type (str, optional): Type of matrix norm to use. Defaults to "nuc".
        read_noise_std (float, optional): Standard deviation of read noise. Defaults to 0.1.
        update_noise_std (float, optional): Standard deviation of update noise. Defaults to 0.1.
        **kwargs: Additional keyword arguments.
    """
    target_weights = self.tile.get_weights()
    self.actual_weight_updates = []
    self.desired_weight_updates = []

    if isinstance(w_init, Tensor):
        self.tile.set_weights(w_init)
    else:
        self.tile.set_weights_uniform_random(-w_init, w_init)

    self.initial_weights = self.tile.get_weights().clone()

    lr_save = self.tile.get_learning_rate()
    self.tile.set_learning_rate(1)

    # mtx_size = self.tile.get_x_size() * self.tile.get_d_size()
    # System matrices
    # A = np.eye(mtx_size)  # State transition matrix
    # B = np.eye(mtx_size)  # Control input matrix
    # C = np.eye(mtx_size)  # Observation matrix

    # Noise covariances
    # Q = np.eye(mtx_size)*update_noise_std**2  # Process noise covariance
    # R = np.eye(mtx_size)*read_noise_std**2  # Measurement noise covariance

    # LQR cost matrices
    # Q_lqr = np.eye(mtx_size)
    # R_lqr = np.zeros_like(R)

    # LQR gain matrix, L = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    # L = np.eye(mtx_size)

    # Initial state estimate
    # S = sla.solve_discrete_are(A.T, C.T, Q, R)
    # If A, C, Q, R are Identity matrices(up to multiplication), Riccati equation reduces to below quadratic equation
    poly = np.array([1, -(update_noise_std**2), -(update_noise_std**2) * (read_noise_std**2)])
    S_scale = np.roots(poly).max()
    # Kalman gain matrix, K = S @ C.T @ np.linalg.inv(C @ S @ C.T + R)
    K_scale = S_scale / (S_scale + read_noise_std)
    x_est = self.tile.get_weights().flatten().numpy()
    for iter in range(max_iter):
        current_weight = self.tile.get_weights().clone()
        # Observation residual, y_res = y - Cx
        z = self.read_weights()[0]
        y_res = z.flatten().numpy() - x_est
        # State estimate, A@x + B@u + K(y - Cx)
        x_est += K_scale * y_res

        # u_vec = -L @ (x_est-target_weights.T.flatten().numpy()) # Control input
        u_vec = -(x_est - target_weights.flatten().numpy())
        # Reshape u into a matrix form
        u_matrix = torch.tensor(
            u_vec.reshape(self.tile.get_d_size(), self.tile.get_x_size()), dtype=torch.float32
        )
        # Perform SVD on the control input, invert the sign due to tile.update() convention
        U, S, Vh = torch.linalg.svd(-u_matrix, full_matrices=False)

        # Use the first singular vector for update
        u = U[:, 0]
        v = Vh[0, :]
        s = S[0]

        # Apply compensation for half-selection
        u1, v1 = compensate_half_selection(u), compensate_half_selection(v)

        # Update the weights
        self.tile.update(v1.float(), u1.float(), False)
        updated_weight = self.tile.get_weights().clone()

        # Realistic weight readout
        diff = z - target_weights
        norm = torch.linalg.matrix_norm(diff, ord=norm_type)
        log.debug(f"Error: {norm}")

        self.actual_weight_updates.append(updated_weight - current_weight)
        self.desired_weight_updates.append(-torch.outer(v1, u1))

        # Check for convergence
        if tolerance is not None and norm < tolerance:
            break

        # Update state estimate, x_est = A @ x_est + B @ u_vec
        x_est += u_vec

    self.tile.set_learning_rate(lr_save)


@torch.no_grad()
def svd_ekf(
    self,
    max_iter: int = 100,
    tolerance: Optional[float] = 0.01,
    w_init: Union[float, Tensor] = 0.0,
    process_noise_std: float = 0.1,
    measurement_noise_std: float = 0.1,
    norm_type: str = "nuc",
    **kwargs: Any,
) -> None:
    target_weights = self.tile.get_weights()

    if isinstance(w_init, Tensor):
        self.tile.set_weights(w_init)
    else:
        self.tile.set_weights_uniform_random(-w_init, w_init)

    lr_save = self.tile.get_learning_rate()
    self.tile.set_learning_rate(1)

    # EKF initialization
    n = target_weights.numel()
    x = self.tile.get_weights().flatten().numpy()
    P = np.eye(n) * 0.2
    R = np.eye(n) * measurement_noise_std**2
    Q = np.eye(n) * process_noise_std**2

    def hx(x):
        return x

    def fx(x, dt):
        return x

    def H_jacobian(x):
        return np.eye(n)

    def F_jacobian(x, dt):
        return np.eye(n)

    ekf = ExtendedKalmanFilter(dim_x=n, dim_z=n)
    ekf.x = x
    ekf.P = P
    ekf.R = R
    ekf.Q = Q
    ekf.H = H_jacobian(x)
    ekf.F = F_jacobian(x, 1)

    for _ in range(max_iter):
        # EKF prediction
        ekf.predict()

        # Get measurement
        z = self.tile.get_weights().flatten().numpy()

        # EKF update
        ekf.update(z, HJacobian=H_jacobian, Hx=hx)

        # Reshape EKF estimate to weight matrix shape
        estimated_weights = ekf.x.reshape(target_weights.shape)

        # Calculate the difference between estimated and target weights
        diff = torch.tensor(estimated_weights) - target_weights

        # Perform SVD on the difference
        U, S, Vh = torch.linalg.svd(diff, full_matrices=False)

        # Use the first singular vectors for rank-1 update
        u = U[:, 0]
        v = Vh[0, :]
        s = S[0]

        # Prepare vectors for update
        sqrt_s = torch.sqrt(s)
        u *= sqrt_s
        v *= sqrt_s
        u1, v1 = compensate_half_selection(u), compensate_half_selection(v)

        # Update tile using rank-1 update
        self.tile.update(v1.float(), u1.float(), False)

        # Check for convergence
        current_weights = self.tile.get_weights()
        norm = torch.linalg.norm(current_weights - target_weights, ord=norm_type)
        log.debug(f"Error: {norm}")
        if tolerance is not None and norm < tolerance:
            break

    self.tile.set_learning_rate(lr_save)  # type: ignore
