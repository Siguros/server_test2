import functools
import math
from typing import Any, Callable, Literal, Sequence, Union

import torch
import torch.nn as nn

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class interleave:
    """Decorator class for interleaving in/out nodes.

    If type is "in", doubles and flips adjacent element for first input tensor. \n If type is
    "out", sum every num_output elems to one for all output tensor. \n If type is "both", does
    both.
    """

    def __init__(self, type: Literal["in", "out", "both"]):
        self.type = type
        # self._num_output = 0
        # self.num_output = interleave._num_output

    def __call__(self, func: Callable[..., torch.Tensor]) -> Callable:
        """Decorator for interleaving in/out nodes."""

        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs) -> torch.Tensor:
            if self.type in ["in", "both"]:
                if len(args) == 1:
                    args = (self.interleave_input(*args),)
                elif len(args) == 2:
                    ins, *others = args
                    args = self.interleave_input(ins), *others
                else:
                    raise ValueError("No input argument found")
            outs = func(obj, *args, **kwargs)
            if self.type in ["out", "both"]:
                outs = self.interleave_output(outs)
            return outs

        return wrapper

    def interleave_output(self, t: torch.Tensor) -> torch.Tensor:
        """Interleave 2D tensor if num output set to even."""
        assert t.dim() == 2, "interleave only works on 2D tensors"
        if self._num_output == 1:
            return t

        elif self._num_output == 2:
            return t[:, ::2] - t[:, 1::2]
        else:
            t_reshaped = t.reshape(t.shape[0], t.shape[1] // self._num_output, self._num_output)
            result = t_reshaped.sum(-1)

            return result

    # @torch.no_grad()
    def interleave_input(self, x: torch.Tensor) -> torch.Tensor:
        """Interleave input tensor."""
        if self._num_input == 1:
            return x.view(x.size(0), -1)
        elif self._num_input == 2:
            x = x.view(x.size(0), -1)  # == x.view(-1,x.size(-1)**2)
            x = x.repeat_interleave(2, dim=1)
            x[:, 1::2] = -x[:, ::2]
            return x
        else:
            raise ValueError("num_input must be 1 or 2")

    @classmethod
    def set_num_input(cls, num_input):
        assert num_input == 2 or num_input == 1, "num_input must be 2 or 1"
        cls._num_input = num_input

    @classmethod
    def set_num_output(cls, num_output):
        assert num_output % 2 == 0 or num_output == 1, "num_output must be even or 1"
        cls._num_output = num_output


class type_as:
    """Decorator class for match output tensor type as input tensor type."""

    def __init__(self, func: Callable[..., Union[Sequence[torch.Tensor], torch.Tensor]]):
        self.func = func
        raise DeprecationWarning("Use torch.Tensor.to(self.device) instead.")

    def __call__(self, obj, *args, **kwargs):
        """Decorator for matching output tensor type as input tensor type."""
        out = self.func(obj, *args, **kwargs)
        if isinstance(out, list):
            return [t.type_as(args[0]) for t in out]
        else:
            return out.type_as(args[0])

    def __get__(self, instance, owner=None):
        return functools.partial(self.__call__, instance)


@torch.jit.script
def deltaV(n: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Compute batch-wise deltaV matrix from 2 node voltages.

    Where deltaV[i, j] = n[i] - m[j]

    Args:
        n (torch.Tensor): (B x) I
        m (torch.Tensor): (B x) O

    Returns:
        torch.Tensor: (B x) O x I
    """
    assert len(n.shape) in [1, 2], "n must be 1D or 2D"
    if len(n.shape) == 2:
        assert n.shape[0] == m.shape[0], "n and m must have the same batch size"
        N = n.clone().unsqueeze(dim=-1).repeat(1, 1, m.shape[-1]).transpose(1, 2)
        M = m.clone().unsqueeze(dim=-1).repeat(1, 1, n.shape[-1])
    else:
        N = n.clone().unsqueeze(dim=-1).repeat(1, m.shape[-1]).T
        M = m.clone().unsqueeze(dim=-1).repeat(1, n.shape[-1])

    return N - M


# Use .apply() for below classes


# TODO: bias 추가
class AdjustParams:
    def __init__(
        self,
        L: Union[float, None] = 1e-7,
        U: Union[float, None] = None,
        clamp: bool = True,
        normalize: bool = False,
    ) -> None:
        self.min = L
        self.max = U
        self.normalize = normalize
        self.clamp = clamp

    @torch.no_grad()
    def __call__(self, submodule: nn.Module):
        """Adjust parameters."""
        for name, param in submodule.named_parameters():
            if name in ["weight", "bias"]:
                if self.clamp:
                    (
                        log.debug(f"Clamping {name}...")
                        if torch.any(param.min() < self.min)
                        else ...
                    )
                    param.clamp_(self.min, self.max)
                if self.normalize:
                    nn.functional.normalize(param, dim=1, p=2)


def positive_param_init(min_w: float = 1e-6, max_w: float | None = None, max_w_gain: float = 0.08):
    """Initialize weights."""
    if max_w is None and max_w_gain is None:
        raise ValueError("Either max_w or max_w_gain must be provided")

    def _init_params(submodule: nn.Module):
        nonlocal min_w, max_w, max_w_gain
        if hasattr(submodule, "weight") and getattr(submodule, "weight") is not None:
            param = submodule.get_parameter("weight")
            # positive xaiver_uniform
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(param)
            upper_thres = max_w_gain / math.sqrt(fan_in + fan_out) if max_w is None else max_w
            nn.init.uniform_(param, min_w, upper_thres)
        if hasattr(submodule, "bias") and getattr(submodule, "bias") is not None:
            nn.init.zeros_(submodule.get_parameter("bias"))

    return _init_params


def gaussian_noise(std: float):
    """Add Gaussian noise to named parameters."""

    def _gaussian_noise(submodule, std: float):
        for _, param in submodule.named_parameters():
            param.data = torch.normal(param.data, std)

    return functools.partial(_gaussian_noise, std=std)
