import numpy as np
from torch.utils.data import Dataset
from typing import Tuple


class Tut_Dataset(Dataset):
    """
    Creates a custom dataset (mandatory step when working with PyTorch)
    that will be used during training.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[index], self.y[index]
