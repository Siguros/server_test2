from abc import ABC
from curses import setupterm
from math import sqrt
from pickletools import optimize
from sys import settrace
from typing import Any, Mapping, Union


class cfgBase(ABC):
    optimizer: str = None
    SPICE_params: dict = None
    dims: list = None
    batch_size: int = None
    num_epochs: int = None
    std_dev: Union[int, float] = None

    def __init__(self):
        pass

    def _attrcheck(self):
        attrs = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        for attr in attrs:
            if getattr(self, attr) is None:
                raise NotImplementedError(f"{attr} is not implemented")

    def to_dict(self):
        return {
            attr: getattr(self, attr)
            for attr in self.__dict__
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        }

    def from_dict(self, d: Mapping[str, Any]):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif k.startswith("SPICE_params"):
                _, *keys = k.split("/")
                self._set_dict_value(self.SPICE_params, list(keys), v)

    @classmethod
    def _set_dict_value(cls, d: dict, keys, value):
        key = keys[0]

        v = d.get(key)
        if v is not None and type(v) is not dict:
            d[key] = value
            return
        else:
            cls._set_dict_value(d[key], keys[1:], value)


class SPICEcfg(cfgBase):
    def __init__(self, dims, **kwargs):
        super().__init__()
        self.SPICE_params = {
            "L": 1e-7,
            "U": [2.8 / sqrt(n + m) for n, m in zip(dims, dims[1:])],
            "A": 4,
            "alpha": [0.1, 0.05],  # ~learning rate
            "beta": 1e-2,
            "Diode": {
                "Path": "/path/to/libraries/diode/switching/1N4148.lib",
                "ModelName": "1N4148",
                "Rectifier": "BidRectifier",
            },
            "noise": 0,  # ratio
        }
        self.mpi_commands = ["mpirun", "-use-hwthread-cpus", "-np", "1", "-cpu-set"]


class miniMNIST16cfg(SPICEcfg):
    INPUT_WIDTH = 16
    INPUT_HEIGHT = 16

    def __init__(self, d: Mapping = None, hidden_dims: list = [100], **kwargs):
        self.num_classes = 5
        self.dims = (
            [self.INPUT_WIDTH * self.INPUT_HEIGHT * 2 + 1] + hidden_dims + [self.num_classes * 2]
        )  # input +-, bias
        super().__init__(dims=self.dims, **kwargs)
        self.optimizer = "adam"
        self.num_epochs = 10
        self.batch_size = 10
        self.optim_kwargs = {}
        self.std_dev = 1
        self.upper_frac = 2.8
        self.frac = 3
        self.from_dict(kwargs)
        self.from_dict(d) if d is not None else None
        self.SPICE_params["U"] = [
            self.upper_frac / sqrt(n + m) for n, m in zip(self.dims, self.dims[1:])
        ]
        self._attrcheck()


class MNISTcfg(SPICEcfg):
    INPUT_WIDTH = 28
    INPUT_HEIGHT = 28

    def __init__(
        self,
        d: Mapping = None,
        hidden_dims: list = [100],
        input_bias: int = 0,
        batch_size: int = 10,
        input_scale: int = 2,
        output_scale: int = 2,
        **kwargs,
    ):
        self.num_classes = 10
        self.dims = (
            [self.INPUT_WIDTH * self.INPUT_HEIGHT * input_scale + input_bias]
            + hidden_dims
            + [self.num_classes * output_scale]
        )  # input +-, bias
        super().__init__(dims=self.dims, **kwargs)
        self.optimizer = "adam"
        self.num_epochs = 10
        self.batch_size = batch_size
        self.optim_kwargs = {}
        self.std_dev = 1
        self.upper_frac = 2.8
        self.frac = 3
        self.from_dict(kwargs)
        self.from_dict(d) if d is not None else None
        self.SPICE_params["U"] = [
            self.upper_frac / sqrt(n + m) for n, m in zip(self.dims, self.dims[1:])
        ]
        self._attrcheck()
