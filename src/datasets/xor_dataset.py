from __future__ import annotations

import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

# create a 2D XOR dataset
@np.vectorize
def fn(x1, x2):
    return 1 if x1 * x2 > 0 else 0

def gen_xor_data(size=1000):
    x1 = np.random.uniform(-1, 1, size)
    x2 = np.random.uniform(-1, 1, size)
    y = fn(x1, x2)
    return np.stack([x1, x2], axis=1), y

class XorDataset(Dataset):

    def __init__(self, X: np.array, y: np.array):
        super().__init__()
        assert len(X) == len(y), "X and y must have the same length"
        self.size = len(X)
        self.X = Tensor(X).float()
        self.y = Tensor(y).float().view(-1, 1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    #overload the __init__
    @classmethod
    def from_csv(cls, filename: str, **kwargs):
        data = pd.read_csv(filename, header=None, **kwargs)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return cls(X, y)

    @classmethod
    def from_random(cls, size:int =5000):
        X, y = gen_xor_data(size)
        return cls(X, y)
    
    def dump(self, filename: str, **kwargs):
        data = zip(self.X, self.y)
        with open(filename, "w") as f:
            for x, y in data:
                f.write(f"{x[0]},{x[1]},{y}\n")
