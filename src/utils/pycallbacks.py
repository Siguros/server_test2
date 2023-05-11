from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, Mapping, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.loggers.base import DummyLogger, LoggerCollection

LOGGER_TYPE = {"tb": TensorBoardLogger, "wandb": WandbLogger}


class AdvLoggerCallback(Callback):
    def __init__(
        self,
        on_step: bool = False,
        on_epoch: bool = True,
        log_options: Dict[str, bool] = {
            "histogram": True,
            "scalar": True,
            "Vdrops": True,
            "minimize": True,
        },
    ):
        super().__init__()
        self.on_step = on_step
        self.on_epoch = on_epoch if not on_step else False
        assert self.on_step or self.on_epoch is True, UserWarning(
            "both on_step and on_epoch are False. Nothing to log"
        )
        self.log_optn = log_options
        self.logger = None

    # def on_init_start(self, trainer: Trainer) -> None:
    #     # return super().on_init_start(trainer)
    #     # further setup required if multiple loggers are used
    #     trainer.logger = self.logger
    @abstractmethod
    def on_sanity_check_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        pass

    @abstractmethod
    def on_train_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        pass

    def on_batch_end(self, trainer: Trainer, pl_module: pl.LightningModule):
        self.log_everything(
            pl_module.backbone, step=trainer.global_step, key_suffix="_batch"
        ) if self.on_step else None

    def on_train_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self.log_everything(pl_module.backbone, step=trainer.current_epoch, key_suffix="_epoch")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self.log_Weights(pl_module.named_parameters(), step=trainer.current_epoch)

    # logging functions for debugging
    @abstractmethod
    def log_histogram(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        pass

    @abstractmethod
    def log_scalars(
        self,
        scalars: Mapping,
        layer_name: str = None,
        step: int = None,
        key_suffix: str = "",
        **kwargs,
    ):
        pass

    @abstractmethod
    def log_plot(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        pass

    # logger independent functions

    def log_Vdrops(
        self,
        Vdrops: Iterable,
        step: int = None,
        key_prefix: str = "",
        key_suffix: str = None,
        **kwargs,
    ):
        if Vdrops is not None:
            for idx, value in enumerate(Vdrops):
                self.log_histogram(f"Vdrop_{key_prefix}_{idx}", value, step, key_suffix, **kwargs)

    # TODO: add log metrics
    def log_everything(
        self, backbone: AnalogEP, key_suffix: str, step: int = None, **kwargs
    ):
        if self.log_optn["Vdrops"]:
            self.log_Vdrops(backbone.fdV, step, key_prefix="free", key_suffix=key_suffix, **kwargs)
            self.log_Vdrops(
                backbone.ndV, step, key_prefix="nudge", key_suffix=key_suffix, **kwargs
            )
        # layerwise
        Weights: Iterator = backbone.named_parameters()
        for key, value in Weights:
            if self.log_optn["histogram"]:
                self.log_histogram(key, value, step, key_suffix, **kwargs)
                self.log_histogram(key + ".grad", value.grad, step, key_suffix, **kwargs)
            if self.log_optn["scalar"]:
                var, mean = torch.var_mean(value)
                var_grad, mean_grad = torch.var_mean(value.grad)
                layer_name, _ = key.split("weight")
                metrics = {
                    layer_name + "var": var,
                    layer_name + "mean": mean,
                    layer_name + "var_grad": var_grad,
                    layer_name + "mean_grad": mean_grad,
                }
                self.log_scalars(metrics, layer_name, step, key_suffix, **kwargs)
        # phasewise
        if self.log_optn["minimize"]:
            handler = backbone.metric_handler
            # for _ in self.num_phases:
            for key, val in handler.metrics.items():
                self.log_plot(key, val, step, key_suffix, **kwargs)
            handler.clear()

    def log_Weights(self, Weights: Generator, step: int = None, **kwargs):
        for key, value in Weights:
            self.log_histogram(key, value, step, **kwargs)

    def log_Weights_norm(self, Weights: Generator, step: int = None, **kwargs):
        norms = dict()
        norms.update({key: value.norm() for key, value in Weights})
        # [norms.update({key:value.norm()}) for key, value in Weights]
        key = "Weights_norm"
        self.log_scalars(norms, key, step, **kwargs)

    def log_Weights_grad(self, Weights: Generator, step: int = None, **kwargs):
        for key, value in Weights:
            key += ".grad"
            self.log_histogram(key, value.grad, step, **kwargs)

    def log_Weights_grad_norm(self, Weights: Generator, step: int = None, **kwargs):
        norms = dict()
        norms.update({key: value.grad.norm() for key, value in Weights})
        # [norms.update({key:value.grad.norm()}) for key, value in Weights]
        key = "Weights_grad_norm"
        self.log_scalars(norms, key, step, **kwargs)


class TBLoggerCallback(AdvLoggerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_sanity_check_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if type(pl_module.logger) is LoggerCollection:
            for logger in pl_module.logger:
                if type(logger) is TensorBoardLogger:
                    self.logger = logger.experiment
                    break
        elif type(pl_module.logger) is TensorBoardLogger:
            self.logger = pl_module.logger.experiment
        else:
            raise ValueError("no tensorboard logger found")

    def on_train_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        self.logger = (
            pl_module.logger.experiment if self.logger is None else self.logger
        )  # for fast_dev_run
        self.log_Weights(pl_module.named_parameters(), step=-1)

    def log_histogram(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        key += key_suffix
        self.logger.add_histogram(key, data, step, **kwargs)

    def log_scalars(
        self,
        scalars: Mapping,
        layer_name: str = None,
        step: int = None,
        key_suffix: str = "",
        **kwargs,
    ):
        for key, value in scalars.items():  # l
            self.logger.add_scalars(layer_name + key_suffix, {key: value}, step, **kwargs)

    def log_plot(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        raise NotImplementedError


import matplotlib.pyplot as plt
import wandb


class WandbLoggerCallback(AdvLoggerCallback):
    # log directly from wandb

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_optn["histogram"] = False
        project = kwargs.get("project", None)
        # wandb.init(project=project)

    def on_sanity_check_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if type(pl_module.logger) is LoggerCollection:
            for logger in pl_module.logger:
                if type(logger) is WandbLogger:
                    self.logger = logger

                    break
        elif type(pl_module.logger) is WandbLogger:
            self.logger = pl_module.logger
        else:
            raise ValueError("W&B logger not found")

    def on_train_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        # assert type(pl_module.logger) is WandbLogger, "W&B logger not found"
        self.logger = WandbLogger() if self.logger is None else self.logger  # for fast_dev_run
        self.logger.watch(pl_module, log="all", log_graph=False)

    def log_histogram(self, key, data, step: int = None, key_suffix: str = "", **kwargs):
        key += key_suffix
        wandb.log({key: wandb.Histogram(data.cpu())}, **kwargs)

    def log_scalars(
        self,
        scalars: Mapping,
        layer_name: str = None,
        step: int = None,
        key_suffix: str = "",
        **kwargs,
    ):
        if key_suffix != "":
            stepsize = key_suffix.strip("_")
            scalars[stepsize] = step
        wandb.log(scalars, **kwargs)

    def log_plot(
        self,
        key,
        data: Iterable[torch.Tensor],
        step: int = None,
        key_suffix: str = "",
        **kwargs,
    ):
        plt.plot(data)
        plt.ylabel(key)
        wandb.log({key: plt}, **kwargs)
        # if kwargs.get('norm', False):
        #     data = torch.tensor(data)
        #     max = data.max().item()
        #     min = data.min().item()
        #     data = [(d.item() - min)/max for d in data]
        # zippeddata = [(idx, datum) for idx, datum in enumerate(data)]
        # table = wandb.Table(data = zippeddata, columns = ["x", "y"])
        # wandb.log({key: wandb.plot.line(table, x='step', y=f'{key}_{step}')}, **kwargs)

    # def log_everything(self, pl_module: pl.LightningModule, key_suffix: str, step: int = None, **kwargs):
    #     #log vdrops additionally
    #     # if self.log_optn['Vdrops']:
    #     #     self.log_Vdrops(pl_module.fdV, step, key_prefix='free',  key_suffix=key_suffix, **kwargs)
    #     #     self.log_Vdrops(pl_module.ndV, step, key_prefix='nudge', key_suffix=key_suffix, **kwargs)
