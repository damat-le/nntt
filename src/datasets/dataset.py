from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

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
        self.X = X
        self.y = y

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

class XorDataModule(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        path: str,
        train_val_test_split: list[int], #l'ho inserito io
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        patch_size: int | Sequence[int] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.path = path
        self.train_val_test_split = train_val_test_split
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str|None = None) -> None:
        data = XorDataset.from_csv(self.path)
        size = data.size

        X, y = data.X, data.y
        
        idx1 = self.train_val_test_split[0]
        idx2 = self.train_val_test_split[1]

        assert idx1 < idx2 < size, "Invalid train/val/test split indices"

        self.train_dataset = XorDataset(X[:idx1, :], y[:idx1])
        self.val_dataset = XorDataset(X[idx1:idx2, :], y[idx1:idx2])
        self.test_dataset = XorDataset(X[idx2:, :], y[idx2:])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            #collate_fn=lambda data: tuple(data)
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
            #collate_fn=lambda data: tuple(data)
        )
    
    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            #collate_fn=lambda data: tuple(data)
        )
     