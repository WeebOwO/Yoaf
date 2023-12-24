import torch
import numpy as np

import lightning as L

from math import log
from config import config

from itertools import product
from torch.utils.data import Dataset
from monai.transforms import apply_transform
from utils.box_utils import corner_to_center
from utils.misc import kl_divergence, collate_v8
from monai.data import DataLoader, load_decathlon_datalist

from .transforms import generate_train_transform, generate_val_transform

class Luna16(Dataset):
    def  __init__(self, data, transform, mode = 'train', max_stride=(4, 4, 4), box_sigma = 0.01, use_rlfa = False) -> None:
        self.data = data
        self.transform = transform
        self.stride = max_stride
        self.box_sigmia = box_sigma    
        self.use_rlfa = use_rlfa
        self.ern = 100
        self.cov_sample = torch.diag(torch.tensor([self.ern] * 3, dtype=torch.float32))
        self.mode = mode
    
    def _add_center_label(self, targets):
        bs = len(targets)
        crop_size_z, crop_size_y, crop_size_x = targets[0]['image'].shape[1:]
        stride_z, stride_y, stride_x = self.stride

        for i in range(bs):
            target = targets[i]

            heatmap_label = torch.zeros([crop_size_z // stride_z, crop_size_y // stride_y, crop_size_x // stride_x], dtype=torch.float32)
            rad_label = torch.zeros([crop_size_z // stride_z, crop_size_y // stride_y, crop_size_x // stride_x], dtype=torch.float32)
            reg_label = torch.zeros([crop_size_z // stride_z, crop_size_y // stride_y, crop_size_x // stride_x, 3], dtype=torch.float32)

            c_boxes = corner_to_center(target['box'])
            
            for box in c_boxes:
                sigma = self.box_sigmia * box[-1]
                down_box = box[:-1] / self.stride
                down_z, down_y, down_x = int(down_box[0]), int(down_box[1]), int(down_box[2])
                heatmap_label[down_z][down_y][down_x] = 1.
                
                origin_z, origin_y, origin_x, origin_d = box
                accurate_rad = origin_d / 2

                min_z, min_y, min_x = max(0, origin_z - accurate_rad) // stride_z , max(0, origin_y - accurate_rad) // stride_y, max(0, origin_x - accurate_rad) // stride_x
                max_z, max_y, max_x = min(crop_size_z - 1, origin_z + accurate_rad) // stride_z, min(crop_size_y - 1, origin_y + accurate_rad) // stride_y, min(crop_size_x - 1, origin_x + accurate_rad) // stride_x
                
                z_range = range(int(min_z), int(max_z + 1))
                y_range = range(int(min_y), int(max_y + 1))
                x_range = range(int(min_x), int(max_x + 1))

                heatmap_label[down_z][down_y][down_x] = 1.
        
                for z_idx, y_idx, x_idx in product(z_range, y_range, x_range):
                    #heatmap_label
                    z_offset, y_offset, x_offect = (down_z - z_idx) * stride_z, (down_y - y_idx) * stride_y, (down_x - x_idx) * stride_x
                    distance = np.sqrt(z_offset * z_offset + y_offset * y_offset + x_offect * x_offect)
                    if distance < accurate_rad / 2:
                        hm_value = 1.0
                    else:
                        # when two nodule generate overlap, we consider the max value
                        hm_value = max(np.exp(-distance / 2 * sigma * sigma), heatmap_label[z_idx][y_idx][x_idx])
                    heatmap_label[z_idx][y_idx][x_idx] = hm_value
                    #reg_bal
                    reg_label[z_idx][y_idx][x_idx][0] = (box[0] - z_idx * stride_z) / stride_z
                    reg_label[z_idx][y_idx][x_idx][1] = (box[1] - y_idx * stride_y) / stride_y
                    reg_label[z_idx][y_idx][x_idx][2] = (box[2] - x_idx * stride_x) / stride_x
                    #rad_label
                    rad_label[z_idx][y_idx][x_idx] = log(box[3] / 2)
                                    
            targets[i]['hm_label'] = heatmap_label
            targets[i]['point_reg_label'] = reg_label
            targets[i]['rad'] = rad_label
    
    def _transform(self, idx):
        data_i = self.data[idx]
        return apply_transform(self.transform, data_i)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        targets = self._transform(index)        
        return targets
            
       
class LunaDataModule(L.LightningDataModule):
    def __init__(self, json_dir, data_dir, num_workers, batch_size, box_sigma, use_rlfa) -> None:
        super().__init__()
        self.json_dir = json_dir
        self.box_sigma = box_sigma
        self.use_rlfa = use_rlfa
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
            
            self.luna_train = Luna16(data=train_data[: int(0.95 * len(train_data))], transform=generate_train_transform(config, batch_size=self.batch_size), box_sigma=self.box_sigma, use_rlfa=self.use_rlfa)
            self.luna_val = Luna16(data=train_data[int(0.95 * len(train_data)) :], transform=generate_val_transform(config), box_sigma=self.box_sigma, use_rlfa=self.use_rlfa)
            
    def train_dataloader(self):
        return DataLoader(dataset=self.luna_train, batch_size=1, shuffle=True, num_workers=self.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_v8)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.luna_val, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_v8)