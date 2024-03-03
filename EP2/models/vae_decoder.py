import torch
import torch.nn.functional as F
from lightning import LightningModule


class VAELitModule(LightningModule):
    def __init__(self, vae, pretrained_vae, optimizer, scheduler):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.vae = vae
        self.vae.encoder = pretrained_vae.encoder
        # loss function
        self.criterion = self.vae_loss

        # Freeze the encoder
        for param in self.vae.encoder.parameters():
            param.requires_grad = False

    def vae_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

    def forward(self, x: torch.Tensor):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        recon_x, mu, logvar = self.vae(x)
        loss = self.criterion(recon_x, x, mu, logvar)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        recon_x, mu, logvar = self.vae(x)
        loss = self.criterion(recon_x, x, mu, logvar)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # Exclude encoder parameters
        params = list(filter(lambda p: p.requires_grad, self.parameters()))

        optimizer = self.hparams.optimizer(params=params)
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
