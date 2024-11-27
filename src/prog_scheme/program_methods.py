from abc import ABC, abstractmethod
from typing import Any, Literal, Optional

import torch
from torch import Tensor
from jaxtyping import Float

from src.core.aihwkit.utils import get_persistent_weights
from src.prog_scheme.kalman import AbstractDeviceFilternCtrl, BaseDeviceEKF, DeviceKF, NoFilter
from src.utils.pylogger import RankedLogger

log = RankedLogger(rank_zero_only=True)

NormType = Literal["nuc", "fro", "inf", "1", "-inf", "2"]  # codespell:ignore fro


class AbstractProgramMethods(ABC):
    """Abstract class for programming methods."""

    @staticmethod
    def init_setup(atile, w_init: float | Tensor) -> None:
        """Initializes the tile with the given initial weights & save lr."""

        if isinstance(w_init, Tensor):
            atile.tile.set_weights(w_init)
        else:
            atile.tile.set_weights_uniform_random(-w_init, w_init)

        atile._initial_weights = get_persistent_weights(atile.tile)

        atile.lr_save = atile.tile.get_learning_rate()
        atile.tile.set_learning_rate(1)

    @staticmethod
    def read_weights_(
        self,
        apply_weight_scaling: bool = False,
        x_values: Float[Tensor, "batch in"] | None = None,  # noqa: F722
        x_rand: bool = False,
        over_sampling: int = 10,
        input_ratio: float = 1.0,
    ) -> tuple[Tensor, Tensor | None]:
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

            input_ratio: Ratio of input neurons to use for the estimation

        Returns:
            a tuple where the first item is the ``[out_size, in_size]`` weight
            matrix; and the second item is either the ``[out_size]`` bias vector
            or ``None`` if the tile is set not to use bias.

        Raises:
            TileError: in case wrong code usage of TileWithPeriphery
        """
        dtype = self.get_dtype()
        if x_values is None:
            batch_size = round(self.in_size * input_ratio)
            x_values = torch.eye(batch_size, self.in_size, device=self.device, dtype=dtype)
            if x_rand:
                x_values = torch.rand(
                    batch_size,
                    self.in_size,
                    device=self.device,
                    dtype=dtype,
                )
        else:
            x_values = x_values.to(self.device)

        x_values = x_values.expand(over_sampling, batch_size, self.in_size).reshape(
            over_sampling * batch_size, self.in_size
        )

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

    @abstractmethod
    def program_weights(
        cls,
        atile,
        fnc,
        batch_size: int = 1,
        learning_rate: float = 1,
        max_iter: int = 100,
        tolerance: float | None = 0.01,
        w_init: float | Tensor = 0.0,
        norm_type: NormType = "nuc",
        x_rand: bool = False,
        over_sampling: int = 10,
        **kwargs: Any,
    ) -> None:
        """Program the target weights into the conductances using the pulsed update."""
        pass


class GDP(AbstractProgramMethods):
    """Program the target weights into the conductances using the pulse update defined."""

    @classmethod
    def program_weights(
        cls,
        atile,
        fnc,
        batch_size: int = 1,
        learning_rate: float = 1,
        max_iter: int = 100,
        tolerance: float | None = 0.01,
        w_init: float | Tensor = 0.0,
        norm_type: NormType = "nuc",
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
        output_size = atile.tile.get_d_size()
        state_size = input_size * output_size

        if fnc is None:
            f = NoFilter(state_size)
        else:
            f = fnc

        f.x_est = atile.tile.get_weights().clone().flatten().detach().numpy()  # Initialize x_est

        x = torch.zeros(batch_size, input_size).to(atile.device)
        for i in range(max_iter):
            start_idx = i * batch_size  # Current batch start index

            # Set row and column indices
            row_indices = torch.arange(batch_size)
            col_indices = (start_idx + row_indices) % input_size

            # Set the corresponding indices in x
            if x_rand:
                x = torch.rand(batch_size, input_size).to(atile.device)
            else:
                x[row_indices, col_indices] = 1
            target = x @ atile.target_weights.T

            y = atile.tile.forward(x.repeat(over_sampling, 1), False)
            # mean over oversampling dim
            y_mean = y.view(over_sampling, batch_size, output_size).mean(dim=0)

            # Calculate error and norm
            if fnc is not None:
                z = (
                    torch.linalg.lstsq(x.repeat(over_sampling, 1), y)
                    .solution.flatten()
                    .detach()
                    .numpy()
                )
                f.update(z)
                W_est = f.x_est.reshape(output_size, input_size)
                y_est = x @ W_est.T
                error = y_est - target
            else:
                error = y_mean - target

            current_weights = get_persistent_weights(atile.tile)
            mtx_diff = current_weights - atile.target_weights
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


class SVD(AbstractProgramMethods):
    """Perform singular value decomposition (SVD) based weight programming."""

    @classmethod
    def program_weights(
        cls,
        atile,
        fnc: AbstractDeviceFilternCtrl | None = None,
        batch_size: int = 1,
        max_iter: int = 100,
        tolerance: float | None = 0.01,
        w_init: float | Tensor = 0.0,
        rank_atol: float | None = 1e-2,
        svd_every_k_iter: int = 1,
        norm_type: NormType = "nuc",
        over_sampling: int = 10,
        x_rand: bool = False,
        **kwargs: Any,
    ) -> None:
        """Perform singular value decomposition (SVD) based weight programming.

        Args:
            atile: The analog tile instance.
            fnc: Instance of Kalman Filter or Extended Kalman Filter.
            max_iter: Maximum number of iterations.
            use_rank_as_criterion: Use rank as stopping criterion.
            tolerance: Tolerance for convergence.
            w_init: Initial value for weights.
            rank_atol: Absolute tolerance for numerical rank computation.
            svd_every_k_iter: Perform SVD every k iterations.
            norm_type: Type of matrix norm to use.
            over_sampling: Number of times to over-sample during weight read.
            x_rand: Use random inputs during weight read.
            **kwargs: Additional keyword arguments.
        """
        # Initialize the tile with the given initial weights
        cls.init_setup(atile, w_init)
        input_size = atile.tile.get_x_size()
        output_size = atile.tile.get_d_size()
        state_size = input_size * output_size

        if fnc is None:
            fnc = NoFilter(state_size)

        fnc.x_est = atile.tile.get_weights().clone().flatten().detach().numpy()  # Initialize x_est

        for iter in range(max_iter):
            i = iter % svd_every_k_iter

            # Read the current weights and update the state estimate
            z = (
                cls.read_weights_(atile, over_sampling=over_sampling, x_rand=x_rand)[0]
                .flatten()
                .detach()
                .numpy()
            )
            fnc.update(z)

            if i == 0:
                # Compute the control input
                if isinstance(fnc, NoFilter | DeviceKF):
                    u_vec = -(fnc.x_est - atile.target_weights.flatten().numpy())
                elif isinstance(fnc, BaseDeviceEKF):
                    # L_diag = fnc.get_lqg_gain(u_prev)
                    L_diag = 1
                    u_vec = -L_diag * (fnc.x_est - atile.target_weights.flatten().numpy())

                u_matrix = torch.from_numpy(u_vec).reshape(output_size, input_size).double()
                U, S, Vh = torch.linalg.svd(-u_matrix, full_matrices=False)

            # Perform SVD update
            u_svd = U[:, i]
            v_svd = Vh[i, :]
            s = S[i]
            sqrt_s = torch.sqrt(s)
            v_svd *= sqrt_s
            u_svd *= sqrt_s
            u1, v1 = compensate_half_selection(u_svd), compensate_half_selection(v_svd)
            atile.tile.update(v1.float(), u1.float(), False)
            u_rank1 = -torch.outer(u1, v1)

            # Predict the next state
            fnc.predict(u_rank1.flatten().numpy())

            # Logging and convergence checking
            current_weights = get_persistent_weights(atile.tile)
            norm = torch.linalg.matrix_norm(current_weights - atile.target_weights, ord=norm_type)
            log.debug(f"Error: {norm}")

            if tolerance is not None and norm < tolerance:
                log.debug("Tolerance reached, stopping early.")
                break

            # Recompute SVD if required
            # if (iter + 1) % svd_every_k_iter == 0:
            #     diff_realistic = (
            #         cls.read_weights_(atile, over_sampling=over_sampling, x_rand=x_rand)[0]
            #         - atile.target_weights
            #     )
            #     U, S, Vh = torch.linalg.svd(diff_realistic.double(), full_matrices=False)


@torch.no_grad()
def iterative_compressed(
    self,
    max_iter: int = 100,
    tolerance: float | None = 0.01,
    w_init: float | Tensor = 0.0,
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
