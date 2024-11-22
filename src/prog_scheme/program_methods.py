from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float
from scipy import linalg as sla
from torch import Tensor

from src.core.aihwkit.utils import get_persistent_weights
from src.prog_scheme.kalman import AbstractDeviceFilternCtrl, DeviceKF
from src.utils.pylogger import RankedLogger

log = RankedLogger(rank_zero_only=True)

NormType = Literal["nuc", "fro", "inf", "1", "-inf", "2"]  # codespell:ignore fro


class AbstractProgramMethods(ABC):
    """Abstract class for programming methods."""

    @abstractmethod
    def __init__(self): ...

    @abstractmethod
    def __call__(self) -> None:
        """Program the target weights into the conductances using the pulse update defined."""
        ...

    @staticmethod
    def init_setup(atile, w_init: Union[float, Tensor]) -> None:
        """Initialize the setup for programming methods."""

        if isinstance(w_init, Tensor):
            atile.tile.set_weights(w_init)
        else:
            atile.tile.set_weights_uniform_random(-w_init, w_init)

        atile.initial_weights = get_persistent_weights(atile.tile)

        atile.lr_save = atile.tile.get_learning_rate()
        atile.tile.set_learning_rate(1)

    @staticmethod
    def read_weights_(
        self,
        apply_weight_scaling: bool = False,
        x_values: Optional[Tensor] = None,
        x_rand: bool = False,
        over_sampling: int = 10,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Reads the weights (and biases) in a realistic manner by using the forward pass for
        weights readout.

            If the tile includes digital periphery (e.g. out scaling),
            these will be applied. Thus this weight is the logical
            weights that correspond to the weights in an FP network.

        Note:
            weights are estimated using the ``lstsq`` solver from torch.

        Args:
            apply_weight_scaling: Whether to rescale the given weight matrix
                and populate the digital output scaling factors as
                specified in the configuration
                new ``weight_scaling_omega`` can be given. Note that
                this will overwrite the existing digital out scaling
                factors.

            x_values: Values to use for estimating the matrix. If
                not given, inputs are standard normal vectors.

            over_sampling: If ``x_values`` is not given,
                ``over_sampling * in_size`` random vectors are used
                for the estimation

        Returns:
            a tuple where the first item is the ``[out_size, in_size]`` weight
            matrix; and the second item is either the ``[out_size]`` bias vector
            or ``None`` if the tile is set not to use bias.

        Raises:
            TileError: in case wrong code usage of TileWithPeriphery
        """
        dtype = self.get_dtype()
        if x_values is None:
            x_values = torch.eye(self.in_size, self.in_size, device=self.device, dtype=dtype)
            if x_rand:
                x_values = torch.rand(self.in_size, self.in_size, device=self.device, dtype=dtype)
        else:
            x_values = x_values.to(self.device)

        x_values = x_values.repeat(over_sampling, 1)

        # forward pass in eval mode
        was_training = self.training
        is_indexed = self.is_indexed()
        self.eval()
        if is_indexed:
            self.analog_ctx.set_indexed(False)
        y_values = self.forward(x_values)
        if was_training:
            self.train()
        if is_indexed:
            self.analog_ctx.set_indexed(True)

        if self.bias is not None:
            y_values -= self.bias

        est_weight = torch.linalg.lstsq(x_values, y_values).solution.T.cpu()
        weight, bias = self._separate_weights(est_weight)

        if self.digital_bias:
            bias = self.bias.detach().cpu()

        if not apply_weight_scaling:
            # we de-apply all scales
            alpha = self.get_scales()
            if alpha is not None:
                alpha = alpha.detach().cpu()
                return (weight / alpha.view(-1, 1), bias / alpha if self.analog_bias else bias)
        return weight, bias


class GDP(AbstractProgramMethods):
    """Program the target weights into the conductances using the pulse update defined."""

    @classmethod
    def call_Program_Method(
        cls,
        atile,
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
        """Program the target weights into the conductances using the pulse update defined."""
        # Ensure self has required methods and attributes
        if not hasattr(atile, "init_setup") or not hasattr(atile, "tile"):
            raise AttributeError("Instance must have 'init_setup' and 'tile' attributes")

        # Initialize with the given weight
        cls.init_setup(atile, w_init)
        input_size = atile.tile.get_x_size()

        x = torch.zeros(batch_size, input_size).to(atile.device)
        for i in range(max_iter):
            start_idx = i * batch_size  # Current batch start index

            # Set row and column indices
            row_indices = torch.arange(batch_size)
            col_indices = (start_idx + row_indices) % input_size

            # Set the corresponding indices in x
            x[row_indices, col_indices] = 1
            target = x @ atile.target_weights.T
            yo = []
            for j in range(over_sampling):
                output = atile.tile.forward(x, False)
                yo.append(output)

            # Calculate the average over-sampled outputs
            yo = torch.stack(yo, dim=0)
            y = yo.mean(dim=0)

            # Calculate error and norm
            error = y - target
            mtx_diff = atile.tile.get_weights() - atile.target_weights
            norm = torch.linalg.matrix_norm(mtx_diff, ord=norm_type)
            log.debug(f"Error: {norm}")

            # Optionally break if tolerance is met
            if tolerance is not None and norm < tolerance:
                log.debug("Tolerance reached, stopping early.")
                break

            # Update the tile with the error
            atile.tile.update(x, error, False)  # type: ignore

            # Reset x for the next iteration
            x[row_indices, col_indices] = 0

        # Restore learning rate
        atile.tile.set_learning_rate(atile.lr_save)  # type: ignore

    @classmethod
    def make_callable(cls):
        # 클래스를 호출처럼 동작하게 만드는 메서드
        def callable_method(*args, **kwargs):
            return cls.call_as_classmethod(*args, **kwargs)

        return callable_method


class SVD(AbstractProgramMethods):
    """Perform singular value decomposition (SVD) based weight programming."""

    @classmethod
    def call_Program_Method(
        cls,
        atile,
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
        """
        cls.init_setup(atile, w_init)
        diff_realistic = (
            cls.read_weights_(atile, over_sampling=over_sampling, x_rand=x_rand)[0]
            - atile.target_weights
        )
        U, S, Vh = torch.linalg.svd(diff_realistic.double(), full_matrices=False)
        rank = torch.linalg.matrix_rank(diff_realistic)
        max_iter = min(max_iter, rank) if use_rank_as_criterion else max_iter
        for iter in range(max_iter):

            i = iter % svd_every_k_iter
            u = U[:, i]
            v = Vh[i, :]
            sqrt_s = torch.sqrt(S[i])
            v *= sqrt_s
            u *= sqrt_s
            u1, v1 = compensate_half_selection(u), compensate_half_selection(v)
            atile.tile.update(v1.float(), u1.float(), False)

            current_weights = get_persistent_weights(atile.tile)
            norm = torch.linalg.matrix_norm(current_weights - atile.target_weights, ord=norm_type)
            log.debug(f"Error: {norm}")

            if tolerance is not None and norm < tolerance:
                break
            if (iter + 1) % svd_every_k_iter == 0:
                diff_realistic = (
                    cls.read_weights_(atile, over_sampling=over_sampling, x_rand=x_rand)[0]
                    - atile.target_weights
                )
                U, S, Vh = torch.linalg.svd(diff_realistic.double(), full_matrices=False)

        atile.tile.set_learning_rate(atile.lr_save)  # type: ignore


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
