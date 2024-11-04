import re
import time
from enum import Enum

import torch
from aihwkit.simulator.configs.configs import MappableRPU
from aihwkit.simulator.parameters.helpers import _PrintableMixin
from aihwkit.simulator.tiles.base import BaseTile

from src.utils.logging_utils import LogCapture


def generate_target_weights(input_size: int, output_size: int, rank: int) -> torch.Tensor:
    w_target = torch.randn(input_size, output_size).clamp_(-1, 1)
    if rank < min(w_target.shape):
        u, s, v = torch.svd(w_target)
        w_target = torch.mm(u[:, :rank], torch.mm(torch.diag(s[:rank]), v[:, :rank].t()))
    w_target /= w_target.abs().max()
    return w_target


def extract_error(log_list, prefix: str = "Error: ") -> list:
    err_list = []
    for log in log_list:
        if prefix in log:
            err_list.append(float(re.findall(prefix + r"([0-9.e-]+)", log)[0]))

    return err_list


def rpuconf2dict(rpuconfig: MappableRPU, max_depth=2, current_depth=0) -> dict:
    result = {}
    for key, val in rpuconfig.__dict__.items():
        if type(val) in (float, int, str, bool, type(None)):  # primitive
            result[key] = val
        elif isinstance(val, type):  # class
            result[key] = val.__name__
        elif isinstance(val, Enum):
            result[key] = val.name
        elif isinstance(val, _PrintableMixin):  # instance
            result[key] = (
                rpuconf2dict(val, max_depth, current_depth + 1)
                if current_depth < max_depth
                else str(val)
            )
            result[key]["is_default"] = True if val.__dict__ == val.__class__().__dict__ else False
        else:
            raise ValueError(f"Unknown type {type(val)} for {key}")
    return result


def program_n_log(
    tiles: tuple[BaseTile, ...], target_weight: torch.Tensor, method_kwargs: dict
) -> list[list[float]]:
    err_lists = []
    for tile in tiles:
        with LogCapture() as logc:
            # tile.tile.set_weights(target_weight)
            tile.target_weights = target_weight.clone()
            start = time.time()
            tile.program_weights(**method_kwargs)
            print(f"Programming time: {time.time() - start:.2f}s")
            log = logc.get_log_list()
        err_list = extract_error(log)
        err_lists.append(err_list)
    return err_lists
