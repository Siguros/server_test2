import io
import logging
from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


class LogCapture:

    def __init__(self, logger_name: str = None, level=logging.DEBUG):
        """Capture log messages to a list.

        Args:
            logger_name (str, optional): Name of the logger. Usually the file path under src. Defaults to None.
            level (int, optional): Log level. Defaults to logging.DEBUG.

        Example:
            with LogCapture("src.core.eqprop.strategy") as log_capture:
                logger = log_capture.logger
                logger.info("Hello")
                logger.info("World")
                log_list = log_capture.get_log_list()
        """
        self.log_stream = io.StringIO()
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        self.stream_handler = logging.StreamHandler(self.log_stream)
        self.formatter = logging.Formatter("%(message)s")
        self.stream_handler.setFormatter(self.formatter)

    def __enter__(self):
        """Add the stream handler to the logger."""
        self.logger.addHandler(self.stream_handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove the stream handler from the logger."""
        self.logger.removeHandler(self.stream_handler)
        self.log_stream.close()

    def get_log_list(self):
        """Get the log messages as a list.

        Note that torch.set_printoptions() will affect the output precision.
        """
        return self.log_stream.getvalue().strip().split("\n")
