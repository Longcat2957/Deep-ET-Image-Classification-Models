import os
from typing import List, Union, Tuple
import torch
from torch.utils.data import Dataset

class ImageNet_1k_Dataset(Dataset):
    def __init__(self, root:str, preload:bool=True):
        super().__init__()
        pass
    
    def __len__(self) -> int:
        pass
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass
