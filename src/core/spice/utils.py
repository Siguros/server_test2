import os

import numpy as np
import torch
from PySpice.Spice.Xyce.RawFile import RawFile
from torch.nn.modules import ModuleList

from . import shallowcircuit


class SPICEParser:
    """Parse data between MyCircuit <-> netlist for Xyce."""

    def updateWeight(netlist: shallowcircuit, W: ModuleList):  # resetState
        """Append Rarray subcircuit to netlist corresponding to updated weight."""
        Rarrays = [
            key for key in netlist.raw_subcircuits.keys() if key.endswith("_resistor_array")
        ]
        for idx, key in enumerate(Rarrays):
            last = True if idx == len(Rarrays) - 1 else False
            netlist.raw_subcircuits[key] = SPICEParser.genRarray(W[idx].weight.data, idx, last)

    def clampLayer(netlist: shallowcircuit, x: torch.Tensor):
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

    def releaseLayer(netlist: shallowcircuit, ygrad):
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

    def fastRawfileParser(raw_file: RawFile, nodenames: tuple, dimensions: list):
        """Fast parser for rawfile."""
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
