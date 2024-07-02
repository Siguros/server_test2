import os
from collections import OrderedDict

from PySpice.Spice.Netlist import Circuit, FixedPinElement


class SimpleElement:
    """Simple elements."""

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


class MyCircuit:
    """_summary_
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
