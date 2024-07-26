from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from typing import Sequence
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule


class BaseDataModule(LightningDataModule):
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
        torch_dataset_class: Dataset,
        train_val_test_split: list[int], #l'ho inserito io
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        predict_batch_size: int = 1024,
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
        self.predict_batch_size = predict_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.torch_dataset_class = torch_dataset_class

    def setup(self, stage: str|None = None) -> None:
        
        data = pd.read_csv(self.path, header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        assert len(X) == len(y), "X and y must have the same length"
        size = len(X)
        idxs = np.arange(size)
        np.random.shuffle(idxs)
        X, y = X[idxs], y[idxs]
        
        # data = LatentComparatorDataset.from_pickle(self.path)
        # size = data.size
        # X, y = data.X, data.y
        
        idx1 = self.train_val_test_split[0]
        idx2 = self.train_val_test_split[1]

        assert idx1 < idx2 < size, "Invalid train/val/test split indices"

        self.train_dataset = self.torch_dataset_class(X[:idx1, :], y[:idx1])
        self.val_dataset = self.torch_dataset_class(X[idx1:idx2, :], y[idx1:idx2])
        self.test_dataset = self.torch_dataset_class(X[idx2:, :], y[idx2:])
        self.predict_dataset = self.torch_dataset_class(X, y)


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
     
    def predict_dataloader(self) -> DataLoader | list[DataLoader]:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.predict_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
            #collate_fn=lambda data: tuple(data)
        )