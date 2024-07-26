from __future__ import annotations
import pytorch_lightning as pl
from torch import Tensor
from torch import optim

import warnings
warnings.filterwarnings("ignore")


class BaseExp(pl.LightningModule):

    def __init__(self, torch_model, params: dict) -> None:
        super().__init__()
        self.model = torch_model
        self.params = params
        #self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass
        self.save_hyperparameters(ignore="torch_model")

    def forward(self, *args, **kwargs) -> Tensor:
        return self.model.forward(
            *args,
            **kwargs
        )

    def training_step(self, batch, batch_idx):
        out_ = self.forward(*batch)
        train_loss = self.model.loss_function(
            *out_,
            batch_idx = batch_idx
        )
        self.log_dict(
            train_loss, 
            sync_dist=True
        )
        return train_loss

    def validation_step(self, batch, batch_idx):
        out_ = self.forward(*batch)
        val_loss = self.model.loss_function(
            *out_,
            batch_idx = batch_idx
            )
        self.log_dict(
            {'val_'+k : v for k, v in val_loss.items()}, 
            sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        out_ = self.forward(*batch)
        test_loss = self.model.loss_function(
            *out_,
            batch_idx = batch_idx
        )
        self.log_dict(
            {'test_'+k : v for k, v in test_loss.items()}, 
            sync_dist=True
        )

    def predict_step(self, batch, batch_idx):
        out_ = self.forward(*batch)
        return out_

    def on_validation_end(self) -> None:
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
        )
        if self.params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma = self.params['scheduler_gamma']
            )
            return [optimizer], [scheduler]
        else: 
            return optimizer
