import gc
from typing import Any, Optional

import torch
import torch.nn.functional as F

from src._eqprop.eqprop_backbone import AnalogEP2
from src.core.eqprop import eqprop_util
from src.models.classifier_module import BinaryClassifierLitModule, ClassifierLitModule


class EqPropLitModule(ClassifierLitModule):
    """EqProp Meta class.

    Manual optimization enabled due to EqProp algorithm.
    Has control over training procedure including weight clipping
    It does not know detailed EqProp algorithm

    A ClassifierLitModule organizes your PyTorch code into 6 sections:
        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: AnalogEP2,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        compile: bool = False,
        num_classes: int = 10,
        criterion: type[torch.nn.modules.loss._Loss] = torch.nn.CrossEntropyLoss,
        param_adjuster: Optional[eqprop_util.AdjustParams] = eqprop_util.AdjustParams(),
        gaussian_std: Optional[float] = None,
    ):
        super().__init__(net, optimizer, scheduler, compile, num_classes, criterion)
        self.automatic_optimization = False
        self.param_adjuster = param_adjuster

    def model_backward(self, loss: torch.Tensor, x: torch.Tensor):
        """Backward pass for EqProp.

        Equivalent to loss.backward() & optimizer.step() in PyTorch.
        """
        self.manual_backward(loss)
        self.net.eqprop(x)
        opt = self.optimizers()
        opt.step()
        opt.zero_grad()
        if self.param_adjuster is not None:
            self.net.apply(self.param_adjuster)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.model_backward(loss, batch[0])

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return loss

    def on_training_step_end(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.train_loss.compute())
        elif sch is not None:
            sch.step()
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=False, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int):
        if self.hparams.gaussian_std:
            self.net.apply(eqprop_util.gaussian_noise(std=self.hparams.gaussian_std))
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self, outputs: Any):
        pass


class EqPropMSELitModule(EqPropLitModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = torch.nn.MSELoss(reduction="sum")

    def model_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        yhat = self.forward(x)
        # yhat = F.softmax(logits, dim=1)
        # make y onehot
        y_onehot = F.one_hot(y, num_classes=self.hparams.num_classes).float()
        loss = self.criterion(yhat, y_onehot)
        preds = torch.argmax(yhat, dim=1)
        return loss, preds, y


class EqPropBinaryLitModule(BinaryClassifierLitModule):

    def __init__(
        self,
        net: AnalogEP2,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        compile: bool = False,
        num_classes: int = 10,
        criterion: type[torch.nn.modules.loss._Loss] = torch.nn.BCEWithLogitsLoss,
        param_adjuster: Optional[eqprop_util.AdjustParams] = eqprop_util.AdjustParams(),
        gaussian_std: Optional[float] = None,
    ):
        super().__init__(net, optimizer, scheduler, compile, num_classes, criterion)
        self.automatic_optimization = False
        self.param_adjuster = param_adjuster

    def model_backward(self, loss: torch.Tensor, x: torch.Tensor):
        """Backward pass for EqProp.

        Equivalent to loss.backward() & optimizer.step() in PyTorch.
        """
        self.manual_backward(loss)
        self.net.eqprop(x)
        opt = self.optimizers()
        opt.step()
        opt.zero_grad()
        if self.param_adjuster is not None:
            self.net.apply(self.param_adjuster)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        self.model_backward(loss, batch[0])

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return loss

    def on_training_step_end(self) -> None:
        gc.collect()
        torch.cuda.empty_cache()
