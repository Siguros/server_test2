import torch
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from torch.nn.modules import ModuleList

from .subcircuits import BidAmplifier, BidRectifier, Neuron, Rarray, ReluRectifier


def createCircuit(
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
        circuit.VoltageSource("s" + str(i), "I" + str(i), circuit.gnd, 0)

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
                L=params["L"][i],
                U=params["U"][i],
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
    circuit.subcircuit(
        Rarray(
            inNodes, outNodes, pre1=lastPrefix, pre2="o", L=params["L"][-1], U=params["U"][-1], G=G
        )
    )
    nodes2 = Rarray.genNodes(inNodes, outNodes, lastPrefix, "o")
    circuit.X("R" + str(n_layers - 1), Rarray.genName(inNodes, outNodes), *nodes2)
    # output I sources
    for i in range(outNodes):
        circuit.CurrentSource("s" + str(i), "o" + str(i), circuit.gnd, 0)
    return circuit
