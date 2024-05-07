from PySpice.Spice.Xyce.RawFile import RawFile
from xmlrpc.server import list_public_methods
from torch.nn.modules import ModuleList
from . import MyCircuit
import torch
import os
import numpy as np


class SPICEParser:
    """parse data between MyCircuit <-> netlist for Xyce
    """
    def updateWeight(netlist:MyCircuit, W:ModuleList): #resetState
        """append Rarray subcircuit to netlist corresponding to updated weight
        """
        Rarrays = [key for key in netlist.raw_subcircuits.keys() if key.endswith('_resistor_array')]
        for idx, key in enumerate(Rarrays):
            last = True if idx == len(Rarrays) - 1 else False
            netlist.raw_subcircuits[key] = SPICEParser.genRarray(W[idx].weight.data, idx, last)

    def clampLayer(netlist:MyCircuit, x:torch.Tensor):
        """_summary_

        Args:
            netlist (_type_): _description_
            x (torch.Tensor): _description_
        """
        Vsources = [(key, elem) for key, elem in netlist.raw_elements.items() 
        if key.startswith('Vs')]
        for idx, (key, elem) in enumerate(Vsources):
            elem.value = x[idx].item() if idx != x.size(0) else 1
        netlist.I_enabled = False
        # for key, elem in netlist.raw_elements.items():
        #     if key.startswith('Vs'):
        #         i = int(key[2:])
        #         elem.value = x[i]
        # v = [key, elem for key, elem in netlist.raw_elements.items() 
        # if key.startswith('Vs')]

        
    def releaseLayer(netlist:MyCircuit, ygrad):
        """_summary_

        Args:
            netlist (_type_): _description_
            I (_type_): _description_
        """
        netlist.I_enabled = True
        Isources = [(key, elem) for key, elem in netlist.raw_elements.items() 
        if key.startswith('Is')]
        # - nodes
        for idx, (key, elem) in enumerate(Isources[::2]):
            elem.value = - ygrad[idx].item()
        # + nodes
        for idx, (key, elem) in enumerate(Isources[1::2]):
            elem.value = + ygrad[idx].item()
        

    def rawfileParser(raw_file:RawFile, n_layers):
        nodes = [(node.name, node.as_ndarray()) for node in raw_file.nodes() 
        if node.name.startswith(('I','H','O'))]
        nodes.sort() #H, I, O 
    
        Vin = []
        Vout = []
        for idx in range(n_layers - 1):
            i_pre = 'I' if idx == 0 else 'H'+str(idx)+'_O'
            o_pre = 'O' if idx == n_layers - 2 else 'H'+str(idx+1)+'_I'
            # pred1 = partial(startswith, i_pre)
            
            vin, remainder = partition(startswith,i_pre, nodes) ####XXX
            vin.sort(key=lambda x: int(x[0][len(i_pre):]))
            _, val = zip(*vin)
            Vin.append(np.array(val).T.flatten())
            # pred2 = partial(startswith, o_pre)
            vout, remainder = partition(startswith, o_pre, remainder)
            vout.sort(key=lambda x: int(x[0][len(o_pre):]))
            _, val2 = zip(*vout)
            Vout.append(np.array(val2).T.flatten())
            nodes = remainder
        return (Vin, Vout)
    
    def fastRawfileParser(raw_file:RawFile, nodenames:tuple, dimensions:list):
        nodes = [(node.name, node.as_ndarray()) for node in raw_file.nodes() 
        if node.name.startswith(('I','H','O'))]
        #sort from I to O
        nodes.sort(key=lambda elem: nodenames.index(elem[0])) #H, I, O 
        _, vals = zip(*nodes)
        vals = list(vals)
        Vin = []
        Vout = []
        for n, m in zip(dimensions, dimensions[1:]):
            Vin.append(np.array(vals[:n]).T.flatten())
            del(vals[:n])
            Vout.append(np.array(vals[:m]).T.flatten())
            del(vals[:m])
        return (Vin, Vout)

    def w_get_Vdrop_w_ypred(voltages):
        Vin, Vout = voltages
        Vdrop = [SPICEParser.deltaV(vi,vo) for vi, vo in zip(Vin, Vout)]
        ypred = Vout[-1][::2]-Vout[-1][1::2]
        ypred = torch.from_numpy(ypred)

        return Vdrop, ypred

    @staticmethod
    def deltaV(arrI, arrO):
        col = arrI.size
        row = arrO.size
        arrI = np.expand_dims(arrI,axis=1)
        arrO = np.expand_dims(arrO,axis=0)
        arrI = arrI.repeat(row, axis=1)
        arrO = arrO.repeat(col, axis=0)
        return arrI-arrO

    @staticmethod
    def genRarray(W: torch.Tensor, idx ,is_last:bool = False):
        #set prefixes
        i_pre = 'I' if idx == 0 else 'H'+str(idx)+'_o'
        o_pre = 'o' if is_last else 'H'+str(idx+1)+'_i'
        # assert Rarray.genName(row, col) is title, 'invalid weight mapping'
        row, col = W.shape #3,9
        # generate nodes
        nodes = list(Rarray.genNodes(col, row, i_pre, o_pre))
        inodes = nodes[:col]
        onodes = nodes[col:]
        # header
        title = Rarray.genName(col, row)
        netlist = '.subckt ' + title + ' ' + ' '.join(nodes) +os.linesep
        # resistors
        for j in range(col): #9
            for i in range(row): #3
                name = 'R' + str(j) + '_' + str(i)
                node = inodes[j] + ' ' + onodes[i]
                netlist += name + ' ' + node + ' ' + str(1/W[i,j].item()) + 'Ohm' + os.linesep
        # footer
        netlist += '.ends ' + title + os.linesep
        return netlist
    
    

class weightClipper(object):
    def __init__(self, L,U):
        self.L = L
        self.U = U
    def __call__(self, modules:ModuleList, param):
        for idx, module in enumerate(modules): 
            if hasattr(module, param):
                lower_lim = self.L[idx] if type(self.L) is list else self.L
                upper_lim = self.U[idx] if type(self.U) is list else self.U
                self.clip(module, param, lower_lim, upper_lim)
    
    def clip(self, module, param, L, U):
        p = getattr(module, param).data
        p = p.clamp(L,U)
        getattr(module, param).data = p

