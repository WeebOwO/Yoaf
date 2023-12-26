import torch
import numpy as np

import lightning as L

from math import log
from config import config

from torch.utils.data import Dataset
from monai.transforms import apply_transform
from utils.misc import collate_v8
from monai.data import DataLoader, load_decathlon_datalist

from .transforms import generate_train_transform, generate_val_transform

class Luna16(Dataset):
    def  __init__(self, data, transform) -> None:
        self.data = data
        self.transform = transform
    
    def _transform(self, idx):
        data_i = self.data[idx]
        return apply_transform(self.transform, data_i)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self._transform(index)        
       
class LunaDataModule(L.LightningDataModule):
    def __init__(self, json_dir, data_dir, num_workers, batch_size) -> None:
        super().__init__()
        self.json_dir = json_dir
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            train_data = load_decathlon_datalist(      
                self.json_dir,
                is_segmentation=True,
                data_list_key="training",
                base_dir=self.data_dir,
            )
            
            self.luna_train = Luna16(data=train_data[: int(0.95 * len(train_data))], transform=generate_train_transform(config, batch_size=self.batch_size))
            self.luna_val = Luna16(data=train_data[int(0.95 * len(train_data)) :], transform=generate_val_transform(config))
            
    def train_dataloader(self):
        return DataLoader(dataset=self.luna_train, batch_size=1, shuffle=True, num_workers=self.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_v8)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.luna_val, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_v8)