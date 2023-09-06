from __future__ import annotations
import pytorch_lightning as pl
from torch import Tensor
from torch import optim
from torchmetrics.classification import BinaryF1Score


import warnings
warnings.filterwarnings("ignore")


class BaseExperiment(pl.LightningModule):

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

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        X, y = batch
        self.curr_device = X.device

        results = self.forward(X, y=y)
        train_loss = self.model.loss_function(
            *results,
            optimizer_idx=optimizer_idx,
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

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        X, y = batch
        self.curr_device = X.device

        results = self.forward(X, y=y)
        val_loss = self.model.loss_function(
            *results,
            optimizer_idx = optimizer_idx,
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

        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params['LR'],
            weight_decay=self.params['weight_decay']
            )
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(
                    getattr(self.model,self.params['submodel']).parameters(),
                    lr=self.params['LR_2']
                    )
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(
                    optims[0],
                    gamma = self.params['scheduler_gamma']
                    )
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(
                            optims[1],
                            gamma = self.params['scheduler_gamma_2']
                            )
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
