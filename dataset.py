import numpy as np
from torch.utils.data import Dataset
from typing import Tuple

class Tut_Dataset(Dataset):
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray) -> None:
        self.X = X
        self.y = y
    
    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self,
                    index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.X[index], self.y[index]