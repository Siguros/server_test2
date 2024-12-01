import re
import time
from typing import Literal

import torch
from aihwkit.simulator.tiles.periphery import TileWithPeriphery

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


def get_program_method_name(tile: TileWithPeriphery) -> Literal["gdp-aihw", "gdp", "svd"]:
    """Get the name of the programming method."""
    name = tile.program_weights.__func__.__qualname__.lower().split(".")[0]
    return name if name != "TileWithPeriphery" else "gdp-aihw"


def program_n_log(
    tile: TileWithPeriphery, target_weight: torch.Tensor, **method_kwargs: dict
) -> list[float]:
    """Program the tile and return the error log."""
    with LogCapture() as logc:
        tile.reference_combined_weights = target_weight.clone()
        start = time.time()
        tile.program_weights(**method_kwargs)
        print(f"Programming time: {time.time() - start:.2f}s")
        log = logc.get_log_list()
    err_list = extract_error(log)
    return err_list
