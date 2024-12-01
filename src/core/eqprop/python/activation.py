import functools
import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class AbstractRectifier(ABC):
    """Base class for rectifiers."""

    def __init__(self, Is, Vth, Vl, Vr):
        self.Is = Is
        self.Vth = Vth
        self.Vl = Vl
        self.Vr = Vr

    @abstractmethod
    def i(self, V: torch.Tensor):
        pass

    @abstractmethod
    def a(self, V: torch.Tensor):
        pass

    @abstractmethod
    def p(self, V: torch.Tensor):
        pass


class IdealRectifier(AbstractRectifier):
    def __init__(self, Vl=-1, Vr=1):
        super().__init__(None, None, Vl, Vr)

    def i(self, V: torch.Tensor):
        return torch.zeros_like(V)

    def a(self, V: torch.Tensor):
        return torch.zeros_like(V)

    @classmethod
    def p(cls, V: torch.Tensor):
        """Compute power."""
        pass


class OTS(AbstractRectifier):
    """Ovonic Threshold Switch rectifier."""

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0.1, Vr=0.9):
        super().__init__(Is, Vth, Vl, Vr)

    def a(self, V: torch.Tensor):
        """Compute admittance.

        Use exponential approximation.
        """
        admittance = (
            self.Is
            / self.Vth
            * (torch.exp((V - self.Vr) / self.Vth) + torch.exp((-V + self.Vl) / self.Vth))
        )
        return admittance

    def i(self, V: torch.Tensor):
        """Compute current."""
        return self.Is * (
            torch.exp((V - self.Vr) / self.Vth) - torch.exp((-V + self.Vl) / self.Vth)
        )

    @classmethod
    def p(cls, V: torch.Tensor):
        """Compute power."""
        pass


class SymOTS(OTS):
    """Symmetric OTS rectifier."""

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0, Vr=0):
        assert Vl == Vr == 0, "Vl and Vr must be 0"
        super().__init__(Is, Vth, Vl, Vr)

    def i_div_a(self, V: torch.Tensor):
        """Compute current/admittance.

        Use exponential approximation.
        """
        return self.Vth * torch.tanh(V / self.Vth)

    def inv_a(self, V: torch.Tensor):
        """Compute inverse of admittance."""
        x = (V - self.Vr) / self.Vth
        abs_x = torch.abs(x)
        return (
            self.Vth / self.Is * torch.exp(abs_x) / (torch.exp(x - abs_x) + torch.exp(-x - abs_x))
        )


class PolyOTS(AbstractRectifier):
    """Polynomial Taylor expansion of OTS rectifier."""

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0.1, Vr=0.9, power=2):
        super().__init__(Is, Vth, Vl, Vr)
        self.power = power

    def a(
        self,
        V: torch.Tensor,
    ):
        """Compute admittance.

        Use polynomial approximation.
        """
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
        """Compute current.

        Use polynomial approximation.
        """
        x1 = (V - self.Vr) / self.Vth
        x2 = (V - self.Vl) / self.Vth
        res = 0
        for i in range(1, self.power + 1):
            res += ((x1 / self.Vth).pow(i) - (-x2 / self.Vth).pow(i)) / math.factorial(i)
        return self.Is * res


class P3OTS(AbstractRectifier):
    """3rd order polynomial OTS rectifier."""

    def __init__(self, Is=1e-8, Vth=0.026, Vl=0.1, Vr=0.9):
        super().__init__(Is, Vth, Vl, Vr)

    def i(self, V: torch.Tensor):
        """Compute current."""
        x = V - (self.Vl + self.Vr) / 2
        return 2 * self.Is / self.Vth * (x.pow(3))

    def a(self, V: torch.Tensor):
        """Compute admittance."""
        x = V - (self.Vl + self.Vr) / 2
        return 2 * self.Is / self.Vth * (3 * x.pow(2))

    def p(self, V: torch.Tensor):
        """Compute power."""
        pass


class SymReLU(AbstractRectifier):
    """Symmetric ReLU rectifier."""

    def __init__(self, Is=1, Vth=1, Vl=-0.5, Vr=0.5):
        super().__init__(Is, Vth, Vl, Vr)

    def i(self, V: torch.Tensor):
        """Compute current."""
        x = V * self.Is
        return ((x - self.Vl) / self.Vth).clamp(max=0) + ((x - self.Vr) / self.Vth).clamp(min=0)

    def a(self, V: torch.Tensor):
        """Compute admittance."""
        x = V * self.Is
        return -((x - self.Vl) / self.Vth < 0).float() + ((x - self.Vr) / self.Vth > 0).float()

    def p(self, V: torch.Tensor):
        """Compute power."""
        pass
