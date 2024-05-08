from operator import add
from typing import Union

import numpy as np
import torch
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import u_Ohm, u_V


# relu bid rectifier
class ReluRectifier(SubCircuitFactory):
    """Relu Rectifier."""

    NAME = "relu_rectifier"
    NODES = "i"

    def __init__(self, diodeModel, V1=0.6 @ u_V, R1=1 @ u_Ohm):
        super().__init__()
        self.X("D1", diodeModel, "n1", "i")
        self.V("in1", "n1", "n2", V1)
        self.R("1", "n2", self.gnd, R1)


# sigmoidal bidirectional rectifier
class BidRectifier(SubCircuitFactory):
    """Bidirectional Rectifier."""

    NAME = "bidirectional_rectifier"
    NODES = "i"

    def __init__(self, diodeModel, V1=0.3 @ u_V, V2=-0.7 @ u_V):
        super().__init__()
        self.X("D1", diodeModel, "i", "n1")
        self.X("D2", diodeModel, "n2", "i")
        self.V("in1", "n1", self.gnd, V1)
        self.V("in2", self.gnd, "n2", V2)


# bidirectional Amplifier
class BidAmplifier(SubCircuitFactory):
    """Bidirectional Amplifier."""

    NAME = "bidirectional_amplifier"
    NODES = ("In", "Out")

    def __init__(self, A=4):
        super().__init__()
        self.VCVS(1, "n1", self.gnd, "In", self.gnd, A)
        self.CCCS(1, "In", self.gnd, "V1", 1 / A)
        self.V(1, "n1", "Out", 0)


# neuron
class Neuron(SubCircuitFactory):
    """Neuron."""

    NAME = "neuron"
    NODES = ("n1", "n2")

    def __init__(
        self,
        Rectifier: Union[str, SubCircuitFactory] = "BidRectifier",
        diodeModel: str = "1N4148",
        A: Union[int, float] = 4,
    ):
        super().__init__()
        if Rectifier == "BidRectifier":
            Rectifier = BidRectifier(diodeModel)
            self.subcircuit(Rectifier)
            self.X(1, "bidirectional_rectifier", "n1")
        elif Rectifier == "ReluRectifier":
            Rectifier = ReluRectifier(diodeModel)
            self.subcircuit(Rectifier)
            self.X(1, "relu_rectifier", "n1")
        else:
            assert isinstance(
                Rectifier, SubCircuitFactory
            ), "Rectifier should be SubCircuitFactory instance"

        self.subcircuit(BidAmplifier(A))
        self.X(2, "bidirectional_amplifier", "n1", "n2")


# n-by-m resistor array
class Rarray(SubCircuitFactory):
    """Resistor Array."""

    NAME = ""
    NODES = tuple()

    def __init__(self, n, m, pre1="i", pre2="o", L=0.0001, U=0.1, G=None):  # 32_13 = 3
        self.NAME = self.genName(n, m)
        self.NODES = self.genNodes(n, m, pre1, pre2)
        super().__init__()
        self.__appendResistor(n, m, L, U, pre1, pre2, G)

    @staticmethod
    def genName(n, m):
        """Generate name for Rarray."""
        return str(n) + "-by-" + str(m) + "_resistor_array"

    @staticmethod
    def genNodes(n, m, prefix1="i", prefix2="o"):
        """Generate nodes for Rarray."""
        nodes = tuple()
        for idx, i in enumerate([n, m]):
            numList = list(range(i))
            numList = list(map(str, numList))
            prefix = [prefix1, prefix2][idx]
            prefixList = [prefix] * i
            ioList = list(map(add, prefixList, numList))
            nodes += tuple(ioList)
        return nodes

    def __appendResistor(self, n, m, L, U, prefix1, prefix2, G):
        """Append resistors to Rarray."""
        if isinstance(G, torch.Tensor):
            # print('G in!')
            G = torch.transpose(G, 0, 1)
            G = G.cpu().numpy()
        elif isinstance(G, np.ndarray):
            pass
        else:
            G = np.random.uniform(low=L, high=U, size=(n, m))
        Rvals = 1 / G
        for i in range(n):
            for j in range(m):
                name = str(i) + "_" + str(j)
                self.R(name, prefix1 + str(i), prefix2 + str(j), Rvals[i, j] @ u_Ohm)
