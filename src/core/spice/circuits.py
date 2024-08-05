import os
from collections import OrderedDict

import torch
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from torch.nn.modules import ModuleList

from .subcircuits import Neuron, Rarray


def create_circuit(
    input: torch.tensor, bias: torch.tensor, W: ModuleList, dimensions: list, **params
):
    """Create pyspice circuit instance using dictionary type parameters.

    params = {   "L" : lower limit of weight of each layer<List/float>,   "U" : upper limit of
    weight of each layer<List/float>,   "A" : amplitude of bidirectional amplifier<float/int>
    "Diode" : {         "Path" : path to diode spice model file<string>,         "ModelName" : name
    of diode model<string>,         "Rectifier" : rectifying subcircuit instance
    <SubcircuitFactory>         }   "alpha" : learning rate of each layer <List/float>   "beta" :
    weight for Cost function <Float> }
    """

    # make circuit
    circuit = Circuit("EPspice")
    # include diode model
    DiodeName = params["Diode"]["ModelName"]
    libraries_path = params["Diode"]["Path"]
    spice_library = SpiceLibrary(libraries_path)
    circuit.include(spice_library[DiodeName])
    input_len = len(input)
    dims = [input_len] + dimensions
    n_layers = len(dims)
    # input V sources
    for i in range(dims[0]):
        circuit.VoltageSource("s" + str(i), "I" + str(i), circuit.gnd, input[i].item())

    # hidden Rarray /neurons   I->H1_i->H1_o // H1_o->H2_i->H2_o ... // Hn-2_o->Hn-1_i->Hn-1_o => {n-2 layers}
    Rectifier = params["Diode"]["Rectifier"]
    circuit.subcircuit(Neuron(Rectifier=Rectifier, diodeModel=DiodeName, A=params["A"]))

    for i in range(n_layers - 2):  # exclude out, input layers: 0~n-2

        inNodes = dims[i]
        outNodes = dims[i + 1]
        Prefix1 = "I" if i == 0 else "H" + str(i) + "_o"  # ith hidden input layer
        Prefix2 = "H" + str(i + 1) + "_i"
        Prefix3 = "H" + str(i + 1) + "_o"
        G = W[i].data
        # print(1/G)
        # Rarray pre1->pre2
        circuit.subcircuit(
            Rarray(
                inNodes,
                outNodes,
                pre1=Prefix1,
                pre2=Prefix2,
                G=G,
            )
        )
        nodes = Rarray.genNodes(inNodes, outNodes, Prefix1, Prefix2)
        circuit.X("R" + str(i + 1), Rarray.genName(inNodes, outNodes), *nodes)
        # Neurons pre2->pre3
        Nname = str(i + 1) + "thN_"
        for j in range(outNodes):
            circuit.X(Nname + str(j), "neuron", Prefix2 + str(j), Prefix3 + str(j))

    # output Rarray
    inNodes = dims[-2]
    outNodes = dims[-1]
    lastPrefix = "H" + str(n_layers - 2) + "_o"
    G = W[-1].data
    circuit.subcircuit(Rarray(inNodes, outNodes, pre1=lastPrefix, pre2="o", G=G))
    nodes2 = Rarray.genNodes(inNodes, outNodes, lastPrefix, "o")
    circuit.X("R" + str(n_layers - 1), Rarray.genName(inNodes, outNodes), *nodes2)
    # output I sources
    for i in range(outNodes):
        circuit.CurrentSource("s" + str(i), "o" + str(i), circuit.gnd, 0)
    return circuit


class SimpleElement:
    """Simple elements that only has node name and dc value."""

    @classmethod
    def copyFromElement(cls, element):
        """Copy from element instance."""
        value = element.dc_value if hasattr(element, "dc_value") else element.subcircuit_name
        instance = cls(element.format_node_names(), value)
        return instance

    def __init__(self, nodes: str, value=None):
        self.nodes = nodes  # name + nodes
        self.value = value

    def __str__(self):
        """Change to string."""
        return self.nodes + " " + str(self.value)


class ShallowCircuit:
    """Contains only top-level elements.

    components:
        subcircuits: string
        elements: OrderedDict
    Returns:
        _type_: _description_
    """

    @classmethod
    def copyFromCircuit(cls, circuit: Circuit):
        """Make a copy from Circuit instance."""
        instance = cls(circuit.title, circuit._includes)
        # copy libs
        # for include in circuit._includes:
        #     instance.include(include)
        # copy elements in main circuit
        # for elem in circuit.elements:
        #     instance._add_element(elem)
        # cast elements to SimpleElement
        for elem in circuit.elements:
            instance.raw_elements[elem.name] = SimpleElement.copyFromElement(elem)
        # cast subcircuits to string and copy
        for key, val in zip(circuit.subcircuit_names, circuit.subcircuits):
            instance.raw_subcircuits[key] = str(val)
        instance._appendnodes()
        return instance

    def __init__(self, title, includes, options=None):
        self.raw_subcircuits = OrderedDict()
        self.raw_elements = OrderedDict()
        self.header = ".title " + title + os.linesep + self.appendlist(".include ", includes)
        self.footer = (
            ".options TEMP = 25\n.options TNOM = 25\n.op\n.end" if options is None else options
        )
        self.I_enabled = False
        self.nodes = tuple()

    def __str__(self):
        """Change to string."""
        # reset raw_spice
        self.raw_spice = ""
        self.subckt2spice()
        # header
        netlist = self.header
        # subcircuits
        netlist += self.raw_spice
        # raw_elements
        # if netlist clamped, float all I sources
        if self.I_enabled:
            for elem in self.raw_elements.values():
                netlist += str(elem) + os.linesep
        else:
            for name, elem in self.raw_elements.items():
                if not name.startswith("Is"):
                    netlist += str(elem) + os.linesep
        # netlist = super().str()
        # footer
        netlist += self.footer
        return netlist

    @staticmethod
    def appendlist(prefix, libs):
        """Append list of libraries to netlist."""
        lines = ""
        for i in libs:
            lines += prefix + str(i) + os.linesep
        return lines

    def subckt2spice(self):
        """Cast subckts to string."""
        for val in self.raw_subcircuits.values():
            self.raw_spice += val

    def _appendnodes(self):
        """Append nodes to netlist."""
        Rarrays = [
            self.raw_elements[key].nodes
            for key in self.raw_elements.keys()
            if key.startswith("XR")
        ]
        for string in Rarrays:
            self.nodes += tuple(string.upper().split(" ")[1:])
