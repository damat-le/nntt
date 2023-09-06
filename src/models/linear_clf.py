from __future__ import annotations

from torch import nn
from torch import Tensor

class LinearClf(nn.Module):

    def __init__(self, **kwargs) -> None:
        
        super().__init__()
        self.clf = nn.Sequential(
                nn.Linear(2, 1024),
                nn.Linear(1024, 1024),
                nn.Linear(1024, 1024),
                nn.Linear(1024, 1024),
                nn.Linear(1024, 1024),
                nn.Linear(1024, 1024),
                nn.Linear(1024, 1024),
                nn.Linear(1024, 1),
                nn.Sigmoid()
        )

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        pred = self.clf(x).float()
        y = kwargs['y'].reshape(-1, 1).float()
        return pred, y
    
    def loss_function(self, *args, **kwargs) -> Tensor:
        pred, y = args
        loss = nn.BCELoss()(pred, y)
        return loss