import gc
from typing import Any

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.eqprop import eqprop_util
from src.models.components.eqprop_backbone import AnalogEP2


class EqPropLitModule(LightningModule):
    """EqProp Meta class.

    Has control over in/output node manipulation and training procedure
    It does not know detailed EqProp algorithm

    A LightningModule organizes your PyTorch code into 6 sections:
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
        scheduler: torch.optim.lr_scheduler,
        double_input: bool = False,
        double_output: bool = False,
        positive_w: bool = False,
        bias: bool = False,
        clip_weights: bool = False,
        normalize_weights: bool = False,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net: AnalogEP2 = net(hyper_params=self.hparams)
        self.net.model.apply(
            eqprop_util.init_params(min_w=1e-5, max_w=1)
        ) if self.hparams.positive_w else ...
        if self.hparams.double_input and self.hparams.double_output:
            eqprop_util.interleave.on()
        elif not self.hparams.double_input and not self.hparams.double_output:
            pass
        else:
            raise ValueError("double_input and double_output must be both True or both False")

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        self.automatic_optimization = False

        # set param clipper
        if self.hparams.clip_weights or self.hparams.normalize_weights:
            self.adjuster = eqprop_util.AdjustParams(
                L=1e-5,
                U=None,
                normalize=self.hparams.normalize_weights,
                clamp=self.hparams.clip_weights,
            )

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def model_forward(self, batch: Any):
        if self.hparams.double_input and self.hparams.double_output:
            self.batch = self.preprocessing_input(batch)
            x, y = self.batch
        elif not self.hparams.double_input and not self.hparams.double_output:
            x, y = batch
            x = x.view(x.shape[0], -1)
            self.batch = (x, y)
        else:
            raise ValueError("double_input and double_output must be both True or both False")
        logits = self.net.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def model_backward(self, loss: torch.Tensor):
        # loss.backward(), execute nudge (+ 3rd) phase
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        self.net.eqprop(self.batch)
        opt.step()
        if self.hparams.clip_weights or self.hparams.normalize_weights:
            self.net.apply(self.adjuster)

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_forward(batch)
        self.model_backward(loss)

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

    # TODO: mem leak?
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_forward(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_forward(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self, outputs: Any):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @staticmethod
    def preprocessing_input(batch):
        x, y = batch
        x = x.view(x.size(0), -1)  # == x.view(-1,x.size(-1)**2)
        x = x.repeat_interleave(2, dim=1)
        x[:, 1::2] = -x[:, ::2]
        return x, y


if __name__ == "__main__":
    _ = EqPropLitModule(None, None, None)
