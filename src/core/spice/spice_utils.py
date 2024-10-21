import os

import numpy as np
import torch
from PySpice.Spice.Xyce.RawFile import RawFile

from . import ShallowCircuit
from .subcircuits import Neuron, Rarray


class SPICENNParser:
    """Parse data between ShallowCircuit <-> netlist."""

    def __init__(self):
        self.circuit = ShallowCircuit()

    def updateWeight(netlist: ShallowCircuit, W: list[torch.Tensor]):  # resetState
        """Append Rarray subcircuit to netlist corresponding to updated weight."""
        Rarrays = [
            key for key in netlist.raw_subcircuits.keys() if key.endswith("_resistor_array")
        ]
        for idx, key in enumerate(Rarrays):
            last = True if idx == len(Rarrays) - 1 else False
            netlist.raw_subcircuits[key] = SPICENNParser.genRarray(W[idx].data, idx, last)

    def clampLayer(netlist: ShallowCircuit, x: torch.Tensor):
        """_summary_

        Args:
            netlist (_type_): _description_
            x (torch.Tensor): _description_
        """
        Vsources = [
            (key, elem) for key, elem in netlist.raw_elements.items() if key.startswith("Vs")
        ]
        for idx, (key, elem) in enumerate(Vsources):
            elem.value = x[idx].item() if idx != x.size(0) else 1
        netlist.I_enabled = False
        # for key, elem in netlist.raw_elements.items():
        #     if key.startswith('Vs'):
        #         i = int(key[2:])
        #         elem.value = x[i]
        # v = [key, elem for key, elem in netlist.raw_elements.items()
        # if key.startswith('Vs')]

    def releaseLayer(netlist: ShallowCircuit, ygrad):
        """_summary_

        Args:
            netlist (_type_): _description_
            I (_type_): _description_
        """
        netlist.I_enabled = True
        Isources = [
            (key, elem) for key, elem in netlist.raw_elements.items() if key.startswith("Is")
        ]
        # - nodes
        for idx, (key, elem) in enumerate(Isources[::2]):
            elem.value = -ygrad[idx].item()
        # + nodes
        for idx, (key, elem) in enumerate(Isources[1::2]):
            elem.value = +ygrad[idx].item()

    def fastRawfileParser(
        raw_file: RawFile, nodenames: tuple, dimensions: list
    ) -> tuple[np.ndarray]:
        """Parse rawfile to extract Vin, Vout from Xyce simulation.

        Returns: (Vin, Vout) where Vin, Vout are list of numpy arrays.
        """
        nodes = [
            (node.name, node.as_ndarray())
            for node in raw_file.nodes()
            if node.name.startswith(("I", "H", "O"))
        ]
        # sort from I to O
        nodes.sort(key=lambda elem: nodenames.index(elem[0]))  # H, I, O
        _, vals = zip(*nodes)
        vals = list(vals)
        Vin = []
        Vout = []
        for n, m in zip(dimensions, dimensions[1:]):
            Vin.append(np.array(vals[:n]).T.flatten())
            del vals[:n]
            Vout.append(np.array(vals[:m]).T.flatten())
            del vals[:m]
        return (Vin, Vout)

    @staticmethod
    def genRarray(W: torch.Tensor, idx, is_last: bool = False):
        # set prefixes
        i_pre = "I" if idx == 0 else "H" + str(idx) + "_o"
        o_pre = "o" if is_last else "H" + str(idx + 1) + "_i"
        # assert Rarray.genName(row, col) is title, 'invalid weight mapping'
        row, col = W.shape  # 3,9
        # generate nodes
        nodes = list(Rarray.genNodes(col, row, i_pre, o_pre))
        inodes = nodes[:col]
        onodes = nodes[col:]
        # header
        title = Rarray.genName(col, row)
        netlist = ".subckt " + title + " " + " ".join(nodes) + os.linesep
        # resistors
        for j in range(col):  # 9
            for i in range(row):  # 3
                name = "R" + str(j) + "_" + str(i)
                node = inodes[j] + " " + onodes[i]
                netlist += name + " " + node + " " + str(1 / W[i, j].item()) + "Ohm" + os.linesep
        # footer
        netlist += ".ends " + title + os.linesep
        return netlist
