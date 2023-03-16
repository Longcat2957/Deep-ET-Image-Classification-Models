import os
import torch
from torch import Tensor
from typing import List, Union, Tuple
from torch.utils.data import Dataset

class ImageNet_1k_Dataset(Dataset):
    def __init__(self, root:str, preload:bool=True):
        super().__init__()
        pass
    
    def __len__(self) -> int:
        pass
    
    def __getitem__(self, index:int) -> Tuple[Tensor, Tensor]:
        pass
