import re
import time

import torch
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


def program_n_log(
    tiles: tuple[BaseTile, ...], target_weight: torch.Tensor, method_kwargs: dict
) -> list[list[float]]:
    err_lists = []
    for tile in tiles:
        with LogCapture() as logc:
            # tile.tile.set_weights(target_weight)
            tile.target_weights = target_weight.clone()
            start = time.time()
            tile.program_weights(tile, **method_kwargs)
            print(f"Programming time: {time.time() - start:.2f}s")
            log = logc.get_log_list()
        err_list = extract_error(log)
        err_lists.append(err_list)
    return err_lists
