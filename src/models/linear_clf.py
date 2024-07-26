from __future__ import annotations

from torch import nn
from torch import Tensor
from torcheval.metrics.functional import binary_f1_score

class LinearClf(nn.Module):

    def __init__(self, **kwargs) -> None:
        
        super().__init__()
        self.clf = nn.Sequential(
                nn.Linear(2, 1024),
                nn.Linear(1024, 1024),
                nn.Linear(1024, 1),
                nn.Sigmoid()
        )
        self.bce_loss = nn.BCELoss()
        self.f1_score = binary_f1_score

    def forward(self,  *args, **kwargs) -> Tensor:
        X, y = args
        pred = self.clf(X)
        return pred, y
    
    def loss_function(self, *args, **kwargs) -> Tensor:
        pred, y = args
        return {
            'loss': self.bce_loss(pred, y),
            'f1_score': self.f1_score(pred.flatten(), y.flatten())
        }
        