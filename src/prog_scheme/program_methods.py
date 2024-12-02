from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch import linalg as tla

from src.core.aihwkit.utils import get_persistent_weights
from src.prog_scheme.filters import AbstractDeviceFilter, NoFilter
from src.prog_scheme.controllers import BaseDeviceController, SVDController
from src.utils.pylogger import RankedLogger
from src.core.aihwkit.types import TileModuleWithPeriphery, NormType
from src.prog_scheme.types import BatchedInput

log = RankedLogger(rank_zero_only=True)


class ProgramMethod(ABC):
    """Abstract class for programming methods."""

    @staticmethod
    def init_setup(atile: TileModuleWithPeriphery, w_init: float | Tensor) -> None:
        """Initializes the tile with the given initial weights & save lr."""
        # atile.actual_weight_updates = []
        # atile.desired_weight_updates = []
        if atile.reference_combined_weights is None:
            atile.reference_combined_weights = atile.tile.get_weights()  # type: ignore

        if isinstance(w_init, Tensor):
            atile.tile.set_weights(w_init)
        else:
            atile.tile.set_weights_uniform_random(-w_init, w_init)

        atile._initial_weights = get_persistent_weights(atile.tile)

        atile.lr_save = atile.tile.get_learning_rate()
        atile.tile.set_learning_rate(1)

    @staticmethod
    def read_weights_(
        atile: TileModuleWithPeriphery,
        apply_weight_scaling: bool = False,
        x_values: BatchedInput | None = None,  # noqa: F722
        x_rand: bool = False,
        over_sampling: int = 10,
        input_ratio: float = 1.0,
    ) -> tuple[Tensor, Tensor | None]:
        """Reads the weights (and biases) in a realistic manner by using the
        forward pass for weights readout.

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
        dtype = atile.get_dtype()
        if x_values is None:
            batch_size = round(atile.in_size * input_ratio)
            x_values = torch.eye(batch_size, atile.in_size, device=atile.device, dtype=dtype)
            if x_rand:
                x_values = torch.rand(
                    batch_size,
                    atile.in_size,
                    device=atile.device,
                    dtype=dtype,
                )
        else:
            x_values = x_values.to(atile.device)
            batch_size = x_values.size(0)

        x_values = x_values.expand(over_sampling, batch_size, -1).reshape(
            over_sampling * batch_size, -1
        )

        # forward pass in eval mode
        was_training = atile.training
        is_indexed = atile.is_indexed()
        atile.eval()
        if is_indexed:
            atile.analog_ctx.set_indexed(False)
        y_values = atile.forward(x_values)
        if was_training:
            atile.train()
        if is_indexed:
            atile.analog_ctx.set_indexed(True)

        if atile.bias is not None:
            y_values -= atile.bias

        est_weight = tla.lstsq(x_values, y_values).solution.T.cpu()
        weight, bias = atile._separate_weights(est_weight)

        if atile.digital_bias:
            bias = atile.bias.detach().cpu()

        if not apply_weight_scaling:
            # we de-apply all scales
            alpha = atile.get_scales()
            if alpha is not None:
                alpha = alpha.detach().cpu()
                return (weight / alpha.view(-1, 1), bias / alpha if atile.analog_bias else bias)
        return weight, bias

    @abstractmethod
    def program_weights(
        atile: TileModuleWithPeriphery,
        filter: AbstractDeviceFilter | None = None,
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
        """Program the target weights into the conductances using the pulsed
        update.

        All the programming methods should implement this method.
        """
        pass


class GDP(ProgramMethod):
    """Modified Gradient Descent Programming (GDP) method."""

    @staticmethod
    def program_weights(
        atile: TileModuleWithPeriphery,
        filter: AbstractDeviceFilter = NoFilter(),
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
        """Program the target weights into the conductances using the pulse
        update method.

        This function programs the target weights into the conductances of the given tile module
        using a pulse update method. The implementation is based on the original method from
        `TileWithPeriphery.program_weights()` in `aihwkit.simulator.tiles.periphery.py`.

        Args:
            atile (TileModuleWithPeriphery): The tile module with periphery to be programmed.
            filter (AbstractDeviceFilter): An optional filter control object for
                estimating and updating weights. Defaults to NoFilter().
            batch_size (int, optional): The size of the batch for each iteration. Defaults to 1.
            learning_rate (float, optional): The learning rate for the update process. Defaults to 1.
            max_iter (int, optional): The maximum number of iterations for the update process. Defaults to 100.
            tolerance (float, optional): The tolerance level for early stopping based on the norm of the weight difference. Defaults to 0.01.
            w_init (float or Tensor, optional): The initial weight value or tensor. Defaults to 0.0.
            norm_type (NormType, optional): The type of norm to use for calculating the weight difference. Defaults to "nuc".
            x_rand (bool, optional): If True, random input vectors are used; otherwise, a deterministic pattern is used. Defaults to False.
            over_sampling (int, optional): The oversampling factor for reading weights. Defaults to 10.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            None
        """

        # Initialize with the given weight
        ProgramMethod.init_setup(atile, w_init)
        input_size = atile.tile.get_x_size()
        output_size = atile.tile.get_d_size()
        state_size = input_size * output_size
        target_weights = atile.reference_combined_weights

        filter.x_est = (
            atile.tile.get_weights().clone().flatten().detach().numpy()
        )  # Initialize x_est

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
            y_target = (target_weights @ x.T).T  # (o,i)@(i,b).T -> b,o
            if isinstance(filter, NoFilter):
                y = atile.tile.forward(x.expand(over_sampling, batch_size, input_size)).mean(0)
                error = y - y_target
            else:
                z = (
                    ProgramMethod.read_weights_(atile, over_sampling=over_sampling, x_values=x)[0]
                    .flatten()
                    .detach()
                    .numpy()
                )
                filter.correct(z)
                # Calculate error and norm
                W_est = filter.get_x_est().reshape(output_size, input_size)  # (o,i)
                y_est = (W_est @ x.T).T  # (o,i)@(i,b).T -> b,o
                error = y_est - y_target

            # Update the tile with the error
            atile.tile.update(x, error, False)  # (b,i), (b,o) -> -(o,i)
            u = -(x.T @ error).T.flatten().numpy()
            filter.predict(u)
            # for i in range(batch_size):
            #     u = -torch.outer(error[i], x[i])  # (o,1) @ (1,i) -> (o,i)
            #     filter.predict(u.flatten().numpy())  # type: ignore

            # Reset x for the next iteration
            x[row_indices, col_indices] = 0

            # Logging and convergence checking
            current_weights = get_persistent_weights(atile.tile)
            mtx_diff = current_weights - target_weights
            # TODO: normalize norm
            norm = tla.matrix_norm(mtx_diff, ord=norm_type)
            log.debug(f"Error: {norm}")

            # Optionally break if tolerance is met
            if tolerance is not None and norm < tolerance:
                log.debug("Tolerance reached, stopping early.")
                break

        # Restore learning rate
        atile.tile.set_learning_rate(atile.lr_save)  # type: ignore


class SVD(ProgramMethod):
    """Perform singular value decomposition (SVD) based weight programming."""

    @staticmethod
    def program_weights(
        atile: TileModuleWithPeriphery,
        filter: AbstractDeviceFilter,
        controller: SVDController,
        max_iter: int = 100,
        tolerance: float | None = 0.01,
        w_init: float | Tensor = 0.0,
        rank_atol: float | None = 1e-2,
        norm_type: NormType = "nuc",
        over_sampling: int = 10,
        x_rand: bool = False,
        input_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Perform singular value decomposition (SVD) based weight programming.

        Args:
            atile: The analog tile instance.
            filter: Instance of Kalman Filter or Extended Kalman Filter.
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
        ProgramMethod.init_setup(atile, w_init)
        input_size = atile.tile.get_x_size()
        output_size = atile.tile.get_d_size()
        state_size = input_size * output_size
        target_weights = atile.reference_combined_weights

        filter.x_est = (
            atile.tile.get_weights().clone().flatten().detach().numpy()
        )  # Initialize x_est

        for iter in range(max_iter):
            # Read the current weights and update the state estimate
            z = (
                ProgramMethod.read_weights_(
                    atile, over_sampling=over_sampling, x_rand=x_rand, input_ratio=input_ratio
                )[0]
                .flatten()
                .detach()
                .numpy()
            )
            filter.correct(z)
            W_est = filter.get_x_est().view(output_size, input_size).double()
            v1, u1 = controller(W_est)
            atile.tile.update(v1.float(), -u1.float(), False)
            u_rank_b = -u1.T @ v1

            # Predict the next state
            filter.predict(u_rank_b.flatten().numpy())

            # Logging and convergence checking
            current_weights = get_persistent_weights(atile.tile)
            norm = tla.matrix_norm(current_weights - target_weights, ord=norm_type)
            log.debug(f"Error: {norm}")

            if tolerance is not None and norm < tolerance:
                log.debug("Tolerance reached, stopping early.")
                break


@torch.no_grad()
def iterative_compressed(
    self,
    max_iter: int = 100,
    tolerance: float | None = 0.01,
    w_init: float | Tensor = 0.0,
    norm_type: NormType = "nuc",
    **kwargs: Any,
) -> None:
    """Iterative weight programming per row. The target weights are programmed
    row by row.

    Args:
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tolerance (Optional[float], optional): Tolerance for convergence. Defaults to 0.01.
        w_init (Union[float, Tensor], optional): Initial value for weights. Defaults to 0.01.
        norm_type (str, optional): Type of matrix norm to use. Defaults to "nuc".
        **kwargs: Additional keyword arguments.
    """
    ProgramMethod.init_setup(self, w_init)
    target_weights = self.reference_combined_weights
    prev_weights = self.initial_weights
    ncol, nrow = self.tile.get_d_size(), self.tile.get_x_size()
    for iter in range(max_iter):
        self.tile.update(target_weights, target_weights - prev_weights, False)
        current_weights = get_persistent_weights(self.tile)
        norm = tla.matrix_norm(current_weights - target_weights, ord=norm_type)
        log.debug(f"Error: {norm}")

        self.actual_weight_updates.append(current_weights - prev_weights)
        self.desired_weight_updates.append(target_weights - current_weights)
        prev_weights = current_weights
        if tolerance is not None and norm < tolerance:
            break

    self.tile.set_learning_rate(self.lr_save)


def compensate_half_selection(v: Tensor) -> Tensor:
    """Compensate the half-selection effect for a vector.

    Args:
        v: Vector to compensate.

    Returns:
        Compensated vector.
    """
    return v
