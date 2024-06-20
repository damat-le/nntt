from __future__ import annotations
import pytorch_lightning as pl
from torch import Tensor
from torch import optim
from torchmetrics.classification import BinaryF1Score


import warnings
warnings.filterwarnings("ignore")


class XorExperiment(pl.LightningModule):

    def __init__(self, torch_model, params: dict) -> None:
        super().__init__()
        self.model = torch_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

        self.f1_metric = BinaryF1Score()

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model.forward(input.float(), **kwargs)

    def training_step(self, batch, batch_idx):
        X, y = batch
        self.curr_device = X.device

        results = self.forward(X, y=y)
        train_loss = self.model.loss_function(
            *results,
            #optimizer_idx=optimizer_idx,
            batch_idx = batch_idx
        )
        
        self.log_dict(
            {
                'loss': train_loss,
                'f1_score': self.f1_metric(*results)
            }, 
            sync_dist=True
        )
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        self.curr_device = X.device

        results = self.forward(X, y=y)
        val_loss = self.model.loss_function(
            *results,
            #optimizer_idx = optimizer_idx,
            batch_idx = batch_idx
            )
        self.log_dict(
            {
                "val_loss": val_loss,
                "val_f1_score": self.f1_metric(*results)
            }, 
            sync_dist=True
        )
        
    def on_validation_end(self) -> None:
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
            )
        return optimizer
