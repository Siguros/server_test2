from typing import Any, Optional, Union

import numpy as np
import torch
from scipy import linalg as sla
from torch import Tensor

from src.core.aihwkit.utils import get_persistent_weights
from src.prog_scheme.kalman import BaseDeviceEKF, BaseDeviceKF
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
    x_rand: bool = False,
    over_sampling: int = 10,
    **kwargs: Any,
) -> None:
    """Program the target weights into the conductances using the pulse update defined.

    Variable batch version of the `program_weights_gdp` method.
    """

    init_setup(self, w_init)
    input_size = self.tile.get_x_size()
    x_values = torch.eye(input_size).to(self.device)

    num_rows = max_iter * batch_size

    if x_rand:
        x_values = torch.rand(num_rows, input_size).to(self.device)
    target_values = x_values @ self.target_weights.to(self.device).T

    target_max = target_values.abs().max().item()
    prev_weights = self.initial_weights
    for i in range(max_iter):

        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        if end_idx > len(x_values):
            # Calculate how much we exceed the length
            exceed_length = end_idx - len(x_values)

            # Slice the arrays and concatenate the exceeded part from the beginning
            x = torch.concatenate((x_values[start_idx:], x_values[:exceed_length]))
            target = torch.concatenate((target_values[start_idx:], target_values[:exceed_length]))
        else:
            x = x_values[start_idx:end_idx]
            target = target_values[start_idx:end_idx]

        yo = []
        for j in range(over_sampling):
            output = self.tile.forward(x, False)
            yo.append(output)

        # 리스트를 새로운 차원으로 스택(stack)
        yo = torch.stack(yo, dim=0)

        # oversampling 차원에 대해 평균 계산
        y = yo.mean(dim=0)
        error = y - target
        err_normalized = error.abs().mean().item() / target_max
        mtx_diff = self.tile.get_weights() - self.target_weights
        norm = torch.linalg.matrix_norm(mtx_diff, ord=norm_type)
        log.debug(f"Error: {norm}")
        # log.debug(f"Error: {err_normalized}")
        """
        if tolerance is not None and norm < tolerance:
            break
        """
        self.tile.update(x, error, False)  # type: ignore
        current_weights = get_persistent_weights(self.tile)
        self.actual_weight_updates.append(current_weights - prev_weights)
        self.desired_weight_updates.append(-error.T @ x)
        prev_weights = current_weights

    self.tile.set_learning_rate(self.lr_save)  # type: ignore


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
    over_sampling: int = 10,
    x_rand: bool = False,
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
    init_setup(self, w_init)
    # x_values = torch.eye(self.tile.get_x_size())
    # x_values = x_values.to(self.device)
    # target_values = x_values @ target_weights.to(self.device).T
    # target_max = target_values.abs().max().item()
    # since tile.update() updates w -= lr*delta_w so flip the sign
    diff_realistic = (
        self.read_weights_(over_sampling=over_sampling, x_rand=x_rand)[0] - self.target_weights
    )
    U, S, Vh = torch.linalg.svd(diff_realistic.double(), full_matrices=False)
    rank = torch.linalg.matrix_rank(diff_realistic)
    # if rank_atol is None:
    #     rank_atol = S.max() * max(diff_realistic.shape) * torch.finfo(S.dtype).eps
    # rank = torch.sum(S > rank_atol).item()
    max_iter = min(max_iter, rank) if use_rank_as_criterion else max_iter
    prev_weights = self.initial_weights
    for iter in range(max_iter):

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

        current_weights = get_persistent_weights(self.tile)
        norm = torch.linalg.matrix_norm(current_weights - self.target_weights, ord=norm_type)
        log.debug(f"Error: {norm}")

        self.actual_weight_updates.append(current_weights - prev_weights)
        self.desired_weight_updates.append(-torch.outer(u1, v1))
        prev_weights = current_weights
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
            diff_realistic = (
                self.read_weights_(over_sampling=over_sampling, x_rand=x_rand)[0]
                - self.target_weights
            )
            U, S, Vh = torch.linalg.svd(diff_realistic.double(), full_matrices=False)

    self.tile.set_learning_rate(self.lr_save)  # type: ignore


def svd_kf(
    self,
    max_iter: int = 100,
    tolerance: Optional[float] = 0.01,
    w_init: Union[float, Tensor] = 0.0,
    norm_type: str = "nuc",
    read_noise_std: float = 0.1,
    update_noise_std: float = 0.1,
    svd_every_k_iter: int = 1,
    over_sampling: int = 10,
    x_rand: bool = False,
    **kwargs: Any,
) -> None:
    """Perform weight programming using SVD and Kalman filter.

    Args:
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tolerance (Optional[float], optional): Tolerance for convergence. Defaults to 0.01.
        w_init (Union[float, Tensor], optional): Initial value for weights. Defaults to 0.01.
        norm_type (str, optional): Type of matrix norm to use. Defaults to "nuc".
        read_noise_std (float, optional): Standard deviation of read noise. Defaults to 0.1.
        update_noise_std (float, optional): Standard deviation of update noise. Defaults to 0.1.
        **kwargs: Additional keyword arguments.
    """
    init_setup(self, w_init)
    state_size = self.tile.get_x_size() * self.tile.get_d_size()
    kf = BaseDeviceKF(state_size, read_noise_std, update_noise_std)
    kf.x_est = self.tile.get_weights().clone().flatten().numpy()
    prev_weights = self.initial_weights
    for iter in range(max_iter):
        i = iter % svd_every_k_iter
        z = self.read_weights_(over_sampling=over_sampling, x_rand=x_rand)[0].flatten().numpy()
        kf.update(z)
        if i == 0:
            u_vec = -(kf.x_est - self.target_weights.flatten().numpy())
            u_matrix = (
                torch.from_numpy(u_vec)
                .reshape(self.tile.get_d_size(), self.tile.get_x_size())
                .double()
            )
            U, S, Vh = torch.linalg.svd(-u_matrix, full_matrices=False)
        u = U[:, i]
        v = Vh[i, :]
        s = S[i]
        sqrt_s = torch.sqrt(s)
        v *= sqrt_s
        u *= sqrt_s
        u1, v1 = compensate_half_selection(u), compensate_half_selection(v)
        self.tile.update(v1.float(), u1.float(), False)
        u_rank1 = -torch.outer(u1, v1)
        kf.predict(u_rank1.flatten().numpy())

        current_weights = get_persistent_weights(self.tile)
        norm = torch.linalg.matrix_norm(current_weights - self.target_weights, ord=norm_type)
        log.debug(f"Error: {norm}")

        self.actual_weight_updates.append(current_weights - prev_weights)
        self.desired_weight_updates.append(u_rank1)
        prev_weights = current_weights
        if tolerance is not None and norm < tolerance:
            break

    self.tile.set_learning_rate(self.lr_save)


def svd_ekf_lqg(
    self,
    device_ekf: BaseDeviceEKF,
    max_iter: int = 100,
    tolerance: Optional[float] = 0.01,
    w_init: Union[float, Tensor] = 0.0,
    norm_type: str = "nuc",
    read_noise_std: float = 0.1,
    update_noise_std: float = 0.1,
    svd_every_k_iter: int = 1,
    over_sampling: int = 10,
    x_rand: bool = False,
    **kwargs: Any,
) -> None:
    """Perform weight programming using Extended Kalman filter. SVD after LQG control is used.

    Args:
        device_ekf (BaseDeviceEKF): Device-aware EKF instance.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tolerance (Optional[float], optional): Tolerance for convergence. Defaults to 0.01.
        w_init (Union[float, Tensor], optional): Initial value for weights. Defaults to 0.01.
        norm_type (str, optional): Type of matrix norm to use. Defaults to "nuc".
        read_noise_std (float, optional): Standard deviation of read noise. Defaults to 0.1.
        update_noise_std (float, optional): Standard deviation of update noise. Defaults to 0.1.
        **kwargs: Additional keyword arguments.
    """
    init_setup(self, w_init)
    state_size = self.tile.get_x_size() * self.tile.get_d_size()
    device_ekf.x_est = self.tile.get_weights().clone().flatten().numpy()
    prev_weights = self.initial_weights
    u_prev = None
    for iter in range(max_iter):
        i = iter % svd_every_k_iter
        z = self.read_weights_(over_sampling=over_sampling, x_rand=x_rand)[0].flatten().numpy()
        device_ekf.update(z)
        if i == 0:
            # L_diag = device_ekf.get_lqg_gain(u_prev)
            L_diag = 1
            u_vec = -L_diag * (device_ekf.x_est - self.target_weights.flatten().numpy())
            u_matrix = (
                torch.from_numpy(u_vec)
                .reshape(self.tile.get_d_size(), self.tile.get_x_size())
                .double()
            )
            U, S, Vh = torch.linalg.svd(-u_matrix, full_matrices=False)
        u_svd = U[:, i]
        v_svd = Vh[i, :]
        s = S[i]
        sqrt_s = torch.sqrt(s)
        v_svd *= sqrt_s
        u_svd *= sqrt_s
        u1, v1 = compensate_half_selection(u_svd), compensate_half_selection(v_svd)
        self.tile.update(v1.float(), u1.float(), False)
        u_rank1 = -torch.outer(u1, v1)
        device_ekf.predict(u_rank1.flatten().numpy())
        u_prev = u_rank1.flatten().numpy()

        current_weights = get_persistent_weights(self.tile)
        norm = torch.linalg.matrix_norm(current_weights - self.target_weights, ord=norm_type)
        log.debug(f"Error: {norm}")

        self.actual_weight_updates.append(current_weights - prev_weights)
        self.desired_weight_updates.append(u_rank1)
        prev_weights = current_weights
        if tolerance is not None and norm < tolerance:
            break

    self.tile.set_learning_rate(self.lr_save)


def init_setup(self, w_init) -> None:
    # self.target_weights = self.tile.get_weights().clone()
    self.actual_weight_updates = []
    self.desired_weight_updates = []

    if isinstance(w_init, Tensor):
        self.tile.set_weights(w_init)
    else:
        self.tile.set_weights_uniform_random(-w_init, w_init)

    self.initial_weights = get_persistent_weights(self.tile)

    self.lr_save = self.tile.get_learning_rate()
    self.tile.set_learning_rate(1)


def compensate_half_selection(v: Tensor) -> Tensor:
    """Compensate the half-selection effect for a vector.

    Args:
        v: Vector to compensate.

    Returns:
        Compensated vector.
    """
    return v
