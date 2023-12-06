from xmlrpc.server import list_public_methods

from torch.nn.modules import ModuleList


class weightClipper:
    def __init__(self, L, U):
        self.L = L
        self.U = U

    def __call__(self, modules: ModuleList, param):
        for idx, module in enumerate(modules):
            if hasattr(module, param):
                lower_lim = self.L[idx] if type(self.L) is list else self.L
                upper_lim = self.U[idx] if type(self.U) is list else self.U
                self.clip(module, param, lower_lim, upper_lim)

    def clip(self, module, param, L, U):
        p = getattr(module, param).data
        p = p.clamp(L, U)
        getattr(module, param).data = p
