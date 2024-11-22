from typing import Any, Literal, Optional, Union

import numpy as np
import torch
from scipy import linalg as sla
from torch import Tensor

from src.core.aihwkit.utils import get_persistent_weights
from src.prog_scheme.kalman import AbstractDeviceFilternCtrl, DeviceKF
from src.utils.pylogger import RankedLogger

log = RankedLogger(rank_zero_only=True)

NormType = Literal["nuc", "fro", "inf", "1", "-inf", "2"]  # codespell:ignore fro


@torch.no_grad()
def iterative_compressed(
    self,
    max_iter: int = 100,
    tolerance: Optional[float] = 0.01,
    w_init: Union[float, Tensor] = 0.0,
    norm_type: NormType = "nuc",
    **kwargs: Any,
) -> None:
    """Iterative weight programming per row. The target weights are programmed row by row.

    Args:
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tolerance (Optional[float], optional): Tolerance for convergence. Defaults to 0.01.
        w_init (Union[float, Tensor], optional): Initial value for weights. Defaults to 0.01.
        norm_type (str, optional): Type of matrix norm to use. Defaults to "nuc".
        **kwargs: Additional keyword arguments.
    """
    init_setup(self, w_init)
    prev_weights = self.initial_weights
    ncol, nrow = self.tile.get_d_size(), self.tile.get_x_size()
    for iter in range(max_iter):
        self.tile.update(self.target_weights, self.target_weights - prev_weights, False)
        current_weights = get_persistent_weights(self.tile)
        norm = torch.linalg.matrix_norm(current_weights - self.target_weights, ord=norm_type)
        log.debug(f"Error: {norm}")

        self.actual_weight_updates.append(current_weights - prev_weights)
        self.desired_weight_updates.append(self.target_weights - current_weights)
        prev_weights = current_weights
        if tolerance is not None and norm < tolerance:
            break

    self.tile.set_learning_rate(self.lr_save)


@torch.no_grad()
def gdp2(
    self,
    batch_size: int = 5,
    learning_rate: float = 1,
    max_iter: int = 100,
    tolerance: Optional[float] = 0.01,
    w_init: Union[float, Tensor] = 0.0,
    norm_type: NormType = "nuc",
    x_rand: bool = False,
    over_sampling: int = 10,
    **kwargs: Any,
) -> None:
    """Program the target weights into the conductances using the pulse update defined.

    Variable batch version of the `program_weights_gdp` method.
    """

    init_setup(self, w_init)
    input_size = self.tile.get_x_size()

    prev_weights = self.initial_weights
    x = torch.zeros(batch_size, input_size).to(self.device)
    for i in range(max_iter):

        start_idx = i * batch_size  # 현재 배치의 시작 인덱스

        # 행(row)과 열(column)의 위치 설정
        row_indices = torch.arange(batch_size)  # k = 0, 1, 2, ..., batch_size-1
        col_indices = (start_idx + row_indices) % input_size  # (start_idx + k) % input_size

        # 해당 위치에 1 설정
        x[row_indices, col_indices] = 1
        target = x @ self.target_weights.T
        yo = []
        for j in range(over_sampling):
            output = self.tile.forward(x, False)
            yo.append(output)

        # 리스트를 새로운 차원으로 스택(stack) ,평균 계산
        yo = torch.stack(yo, dim=0)
        y = yo.mean(dim=0)
        error = y - target
        mtx_diff = self.tile.get_weights() - self.target_weights
        norm = torch.linalg.matrix_norm(mtx_diff, ord=norm_type)
        log.debug(f"Error: {norm}")
        # log.debug(f"Error: {err_normalized}")
        """
        if tolerance is not None and norm < tolerance:
            break
        """
        self.tile.update(x, error, False)  # type: ignore
        x[row_indices, col_indices] = 0  # 다음 반복을 위해 0으로 초기화

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
    norm_type: NormType = "nuc",
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
    norm_type: NormType = "nuc",
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
    kf = DeviceKF(state_size, read_noise_std, update_noise_std)
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
    fnc: Optional[AbstractDeviceFilternCtrl] = None,
    max_iter: int = 100,
    tolerance: Optional[float] = 0.01,
    w_init: Union[float, Tensor] = 0.0,
    norm_type: NormType = "nuc",
    read_noise_std: float = 0.1,
    update_noise_std: float = 0.1,
    svd_every_k_iter: int = 1,
    over_sampling: int = 10,
    x_rand: bool = False,
    **kwargs: Any,
) -> None:
    """Perform weight programming using Extended Kalman filter. SVD after LQG control is used.

    Args:
        fnc (AbstractDeviceFilternCtrl): Device-aware EKF instance.
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
    fnc.x_est = self.tile.get_weights().clone().flatten().numpy()
    prev_weights = self.initial_weights
    u_prev = None
    for iter in range(max_iter):
        i = iter % svd_every_k_iter
        z = self.read_weights_(over_sampling=over_sampling, x_rand=x_rand)[0].flatten().numpy()
        fnc.update(z)
        if i == 0:
            # L_diag = fnc.get_lqg_gain(u_prev)
            L_diag = 1
            u_vec = -L_diag * (fnc.x_est - self.target_weights.flatten().numpy())
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
        fnc.predict(u_rank1.flatten().numpy())
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
