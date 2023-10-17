import functools
import math
from typing import Callable, Sequence, Union

import torch
import torch.nn as nn


class interleave:
    """Decorator class for interleaving in/out nodes."""

    _num_output: int = None

    def __init__(self, type: str = None):
        assert type == "in" or type == "out", 'type must be either "in" or "out"'
        self.type = type
        # self.num_output = interleave._num_output

    def __call__(self, func: Callable[..., Sequence[torch.Tensor]]) -> Callable:
        # print(f"called. , {self._on}")
        @functools.wraps(func)
        def wrapper(obj, *args, **kwargs) -> Sequence[torch.Tensor]:
            if self._num_output:
                if self.type == "in":  # mod y_hat
                    y_hat, *others = args
                    new_args = self.interleave(y_hat), *others
                    outs = func(obj, *new_args, **kwargs)
                    return outs
                elif self.type == "out":  # mod all outputs
                    outs = func(obj, *args, **kwargs)
                    # outs = [self.interleave(out) for out in outs]
                    return self.interleave(outs)
            else:
                return func(obj, *args, **kwargs)

        return wrapper

    def interleave(self, t: torch.Tensor) -> torch.Tensor:
        """Interleave 2D tensor."""
        assert t.dim() == 2, "interleave only works on 2D tensors"
        if self._num_output == 1:
            return t

        elif self._num_output == 2:
            return t[:, ::2] - t[:, 1::2]
        else:
            t_reshaped = t.reshape(t.shape[0], t.shape[1] // self._num_output, self._num_output)
            result = t_reshaped.sum(-1)

            return result

    @classmethod
    def on(cls):
        cls._on = True

    @classmethod
    def set_num_output(cls, num_output):
        assert num_output % 2 == 0, "num_output must be even"
        cls._num_output = num_output


class type_as:
    """Decorator class for match output tensor type as input tensor type."""

    def __init__(self, func: Callable[..., Union[Sequence[torch.Tensor], torch.Tensor]]):
        self.func = func

    def __call__(self, obj, *args, **kwargs):
        out = self.func(obj, *args, **kwargs)
        if type(out) is list:
            return [t.type_as(args[0]) for t in out]
        else:
            return out.type_as(args[0])

    def __get__(self, instance, owner=None):
        return functools.partial(self.__call__, instance)


class BaseRectifier:
    def __init__(self, Is, Vth, Vl, Vr):
        self.Is = Is
        self.Vth = Vth
        self.Vl = Vl
        self.Vr = Vr

    def i(self, V: torch.Tensor):
        raise NotImplementedError

    def a(self, V: torch.Tensor):
        raise NotImplementedError

    def p(self, V: torch.Tensor):
        raise NotImplementedError


class OTS(BaseRectifier):
    def __init__(self, Is=1e-8, Vth=0.026, Vl=0.1, Vr=0.9):
        super().__init__(Is, Vth, Vl, Vr)

    def a(self, V: torch.Tensor):
        admittance = (
            self.Is
            / self.Vth
            * (torch.exp((V - self.Vr) / self.Vth) + torch.exp((-V + self.Vl) / self.Vth))
        )
        return admittance

    def i(self, V: torch.Tensor):
        return self.Is * (
            torch.exp((V - self.Vr) / self.Vth) - torch.exp((-V + self.Vl) / self.Vth)
        )

    @classmethod
    def p(cls, V: torch.Tensor):
        pass


class SymOTS(OTS):
    def __init__(self, Is=1e-8, Vth=0.026, Vl=-0.5, Vr=0.5):
        assert Vl == -Vr, "Vl must be equal to -Vr"
        super().__init__(Is, Vth, Vl, Vr)

    def i_div_a(self, V: torch.Tensor):
        xr = (V - self.Vr) / self.Vth
        xl = (-V + self.Vl) / self.Vth
        # xmax = torch.max(xr, xl)
        return self.Vth * ((torch.exp(xr) - torch.exp(xl)) / (torch.exp(xr) + torch.exp(xl)))

    def inv_a(self, V: torch.Tensor):
        x = (V - self.Vr) / self.Vth
        abs_x = torch.abs(x)
        return (
            self.Vth / self.Is * torch.exp(abs_x) / (torch.exp(x - abs_x) + torch.exp(-x - abs_x))
        )


class PolyOTS(BaseRectifier):
    """Polynomial Taylor expansion of OTS rectifier."""

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0.1, Vr=0.9, power=2):
        super().__init__(Is, Vth, Vl, Vr)
        self.power = power

    def a(
        self,
        V: torch.Tensor,
    ):
        x1 = (V - self.Vr) / self.Vth
        x2 = (V - self.Vl) / self.Vth
        res = 2
        for i in range(1, self.power + 1):
            res += i * (x1.pow(i) - (-x2).pow(i)) / math.factorial(i)
        return self.Is / (self.Vth**2) * res

    def i(
        self,
        V: torch.Tensor,
    ):
        x1 = (V - self.Vr) / self.Vth
        x2 = (V - self.Vl) / self.Vth
        res = 0
        for i in range(1, self.power + 1):
            res += ((x1 / self.Vth).pow(i) - (-x2 / self.Vth).pow(i)) / math.factorial(i)
        return self.Is * res


class P3OTS(BaseRectifier):
    """3rd order polynomial OTS rectifier."""

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0.1, Vr=0.9):
        super().__init__(Is, Vth, Vl, Vr)

    def i(self, V: torch.Tensor):
        x = V - (self.Vl + self.Vr) / 2
        return 2 * self.Is / self.Vth * (x.pow(3))

    def a(self, V: torch.Tensor):
        x = V - (self.Vl + self.Vr) / 2
        return 2 * self.Is / self.Vth * (3 * x.pow(2))


class SymReLU(BaseRectifier):
    """Symmetric ReLU rectifier."""

    def __init__(self, Is=1, Vth=1, Vl=-0.5, Vr=0.5):
        super().__init__(Is, Vth, Vl, Vr)

    def i(self, V: torch.Tensor):
        x = V * self.Is
        return ((x - self.Vl) / self.Vth).clamp(max=0) + ((x - self.Vr) / self.Vth).clamp(min=0)

    def a(self, V: torch.Tensor):
        x = V * self.Is
        return -((x - self.Vl) / self.Vth < 0).float() + ((x - self.Vr) / self.Vth > 0).float()


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
class AddNodes:
    # CNN, RNN이면?
    def __init__(self, input_size: torch.Size):
        raise DeprecationWarning("Use method self.init_nodes instead")
        self.layerinput = torch.zeros(input_size)

    def __call__(self, submodule: nn.Module):
        if hasattr(submodule, "weight"):
            assert submodule._get_name() in ["Linear"], "Only Linear layer is supported"
            self.layerinput: torch.Tensor = submodule(self.layerinput)
            shape_node = self.layerinput.shape
            free_node = torch.zeros(shape_node)
            nudge_node = torch.zeros(shape_node)
            submodule.register_buffer("free_node", free_node)
            submodule.register_buffer("nudge_node", nudge_node)


# TODO: bias 추가
class AdjustParams:
    def __init__(
        self,
        L: Union[float, None] = 0.0,
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
        for name, param in submodule.named_parameters():
            if name == "weight":
                if self.clamp:
                    param.clamp_(self.min, self.max)
                if self.normalize:
                    nn.functional.normalize(param, dim=1, p=2)


def _init_params(
    submodule: nn.Module,
    param_name: str = "weight",
    min_w: float = 1e-6,
    max_w_gain: float = 1,
):
    if hasattr(submodule, param_name):
        param = submodule.get_parameter(param_name)
        # positive xaiver_uniform
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(param)
        max_w = max_w_gain / math.sqrt(fan_in + fan_out)
        nn.init.uniform_(param, min_w, max_w)


def init_params(min_w: float = 1e-6, max_w_gain: float = 0.08):
    return functools.partial(_init_params, param_name="weight", min_w=min_w, max_w_gain=max_w_gain)
