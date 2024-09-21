from typing import Any, Optional, Union

import numpy as np
import torch
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter
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
) -> None:
    """Program the target weights into the conductances using the pulse update defined.

    Variable batch version of the `program_weights_gdp` method.
    """

    self.weight_update_log = []
    self.desired_weight_update = []
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

        self.weight_update_log.append(updated_weight - current_weight)
        self.desired_weight_update.append(-torch.einsum("bi, bj -> ij", error, x))
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
    svd_once: bool = False,
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
        rank_rtol (float, optional): Relative tolerance for numerical rank computation. Defaults to 1e-6.
        svd_once (bool, optional): Flag indicating whether to perform SVD only once. Defaults to False.
        norm_type (str, optional): Type of matrix norm to use. Defaults to "nuc".
        **kwargs: Additional keyword arguments.
    Returns:
        None
    """

    self.weight_update_log = []
    self.desired_weight_update = []
    target_weights = self.tile.get_weights()
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
    # normalize diff matrix
    U, S, Vh = torch.linalg.svd(diff.double(), full_matrices=False)
    # rank = torch.linalg.matrix_rank(diff)
    if rank_atol is None:
        rank_atol = S.max() * max(diff.shape) * torch.finfo(S.dtype).eps

    rank = torch.sum(S > rank_atol).item()
    i = 0
    max_iter = min(max_iter, rank) if use_rank_as_criterion else max_iter
    for _ in range(max_iter):
        current_weight = self.tile.get_weights().clone()
        u = U[:, i]
        v = Vh[i, :]
        # uv_ratio = torch.sqrt(u/v)
        uv_ratio = 1
        sqrt_s = torch.sqrt(S[i])
        v *= uv_ratio * sqrt_s
        u *= sqrt_s / uv_ratio
        u1, v1 = compensate_half_selection(u), compensate_half_selection(v)
        self.tile.update(v1.float(), u1.float(), False)

        # TODO: self.get_weights()
        diff = self.read_weights()[0] - target_weights
        U, S, Vh = torch.linalg.svd(diff.double(), full_matrices=False)
        norm = torch.linalg.matrix_norm(diff, ord=norm_type)
        log.debug(f"Error: {norm}")
        # y = self.tile.forward(x_values, False)
        # # TODO: error와 weight 2norm 사이 관계 분석
        # error = y - target_values
        # err_normalized = error.abs().mean().item() / target_max
        # log.debug(f"Error: {err_normalized}")
        updated_weight = self.tile.get_weights().clone()
        self.weight_update_log.append(updated_weight - current_weight)
        # self.desired_weight_update.append(-torch.einsum('bi, bj -> ij', u1, v1))
        if tolerance is not None and norm < tolerance:
            break
        elif svd_once:
            i += 1
        else:
            pass

    self.tile.set_learning_rate(lr_save)  # type: ignore


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

    def f(x):
        return x

    def h(x):
        return self.tile.get_weights().flatten().numpy()

    for _ in range(max_iter):
        # EKF prediction
        x = f(x)
        P = P + Q

        # Get measurement
        z = self.tile.get_weights().flatten().numpy()

        # EKF update
        y = z - h(x)
        H = np.eye(n)  # Jacobian of h(x) with respect to x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(n) - K @ H) @ P

        # Reshape EKF estimate to weight matrix shape
        estimated_weights = x.reshape(target_weights.shape)

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
        # y = self.tile.forward(x_values, False)
        # # TODO: error와 weight 2norm 사이 관계 분석
        # error = y - target_values
        # err_normalized = error.abs().mean().item() / target_max
        # log.debug(f"Error: {err_normalized}")
        if tolerance is not None and norm < tolerance:
            break
        else:
            pass

    self.tile.set_learning_rate(lr_save)  # type: ignore


@torch.no_grad()
def svd_ukf(
    self,
    max_iter: int = 100,
    tolerance: Optional[float] = 0.01,
    w_init: Union[float, Tensor] = 0.0,
    process_noise_std: float = 0.1,
    measurement_noise_std: float = 0.1,
    **kwargs: Any,
) -> None:
    target_weights = self.tile.get_weights()

    if isinstance(w_init, Tensor):
        self.tile.set_weights(w_init)
    else:
        self.tile.set_weights_uniform_random(-w_init, w_init)

    lr_save = self.tile.get_learning_rate()
    self.tile.set_learning_rate(1)

    # UKF initialization
    n = target_weights.numel()
    points = MerweScaledSigmaPoints(n, alpha=0.1, beta=2.0, kappa=-1)

    def f(x, dt):
        return x

    def h(x):
        return self.tile.get_weights().flatten().numpy()

    ukf = UnscentedKalmanFilter(dim_x=n, dim_z=n, dt=1.0, fx=f, hx=h, points=points)
    ukf.x = self.tile.get_weights().flatten().numpy()
    ukf.P *= 0.2
    ukf.R = np.eye(n) * measurement_noise_std**2
    ukf.Q = np.eye(n) * process_noise_std**2

    for _ in range(max_iter):
        # UKF prediction
        ukf.predict()

        # Get measurement
        z = self.tile.get_weights().flatten().numpy()

        # UKF update
        ukf.update(z)

        # Reshape UKF estimate to weight matrix shape
        estimated_weights = ukf.x.reshape(target_weights.shape)

        # Calculate the difference between estimated and target weights
        diff = torch.tensor(estimated_weights) - target_weights

        # Perform SVD on the difference
        U, S, Vh = torch.linalg.svd(diff.double(), full_matrices=False)

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
        nuc_norm = torch.linalg.norm(current_weights - target_weights, ord="nuc")
        log.debug(f"Error: {nuc_norm}")
        # y = self.tile.forward(x_values, False)
        # # TODO: error와 weight 2norm 사이 관계 분석
        # error = y - target_values
        # err_normalized = error.abs().mean().item() / target_max
        # log.debug(f"Error: {err_normalized}")
        if tolerance is not None and nuc_norm < tolerance:
            break
        else:
            pass

    self.tile.set_learning_rate(lr_save)  # type: ignore
