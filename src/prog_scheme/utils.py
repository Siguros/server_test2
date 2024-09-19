import re
from enum import Enum

import torch
from aihwkit.simulator.tiles.base import BaseTile
from matplotlib import pyplot as plt

from src.utils.logging_utils import LogCapture


def plot_singular_values(Ws: tuple[torch.Tensor]):
    for w in Ws:
        s = torch.linalg.svdvals(w)
        plt.plot(s)
    plt.yscale("log")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value")
    plt.title("Singular Values of Weight Matrix")
    plt.show()


def extract_error(log_list, prefix: str = "Error: ") -> list:
    err_list = []
    for log in log_list:
        if prefix in log:
            err_list.append(float(re.findall(prefix + r"([0-9.e-]+)", log)[0]))

    return err_list


def rpuconf2dict(rpuconfig, max_depth=2, current_depth=0):
    if current_depth > max_depth:
        return rpuconfig
    result = {}
    for key, val in rpuconfig.__dict__.items():
        if isinstance(val, (float, int, str, bool)):
            result[key] = val
        elif isinstance(val, type):
            result[key] = val.__name__
        elif isinstance(val, Enum):
            result[key] = val.name
        else:
            result[key] = rpuconf2dict(val, max_depth, current_depth + 1)
    return result


def program_n_log(
    tiles: tuple[BaseTile, ...], target_weight: torch.Tensor, method_kwargs: dict
) -> list[list[float]]:
    err_lists = []
    for tile in tiles:
        with LogCapture() as logc:
            tile.tile.set_weights(target_weight)
            tile.program_weights(**method_kwargs)
            log = logc.get_log_list()
        err_list = extract_error(log)
        err_lists.append(err_list)
    return err_lists
