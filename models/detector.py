import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from .heads import *
from .backbone import *

from config import net_config
from utils.box_utils import make_anchors, dist2bbox, nms_3D

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau
    
def build_from_dict(indict : dict):
    class_name = indict['type']
    instance_class = globals()[class_name]
    del indict['type']
    return instance_class(**indict)

class Detector(nn.Module):
    def __init__(self, backbone_dict, head_dict) -> None:
        super().__init__()
        self.backbone = build_from_dict(backbone_dict)
        self.neck = None # todo: add neck to intergrate different scale feature
        self.head = build_from_dict(head_dict)
        self.criterion = None
    
    def set_criterion(self, criterion):
        self.criterion = criterion
            
    def inference(self, x):
        _, _, d, h, w = x.shape
        feature = self.backbone(x)
        cls_feature, reg_feature = self.head(feature)
        anchor_points, stride_tensor = make_anchors(cls_feature, (d, h, w), grid_cell_offset=0.5)
        
        pred_scores = cls_feature.view(cls_feature.shape[0], 1, -1)
        pred_distri = reg_feature.view(reg_feature.shape[0], self.head.reg_max * 6, -1)
        
        anchor_points = anchor_points.transpose(0, 1)
        stride_tensor = stride_tensor.transpose(0, 1)
        
        box = dist2bbox(self.head.dfl(pred_distri), anchor_points.unsqueeze(0), xywh=True, dim=1) * stride_tensor
        probs = pred_scores.sigmoid()
        
        pred = torch.cat((probs, box), 1).transpose(1, 2).squeeze()
        keep = nms_3D(pred)
        return pred[keep]
        
    def forward(self, x):
        inputs, targets = x
        feature = self.backbone(inputs)
        pred = self.head(feature)
    
        return self.criterion(pred, targets)

class Model(L.LightningModule):
    def __init__(self, build_dict, train_config=None):
        # record backbone type 
        super().__init__()
        self.detector = Detector(build_dict['backbone'], build_dict['head'])
        
        if train_config != None:
            self.lr = train_config.get('lr', 0)
            self.momentum = train_config.get('momentum', 0)
            self.warm_up = train_config.get('warm_up', 0) 
            self.t_max =  train_config.get('t_max', 0)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.detector.parameters(), self.lr or self.learning_rate, weight_decay=1e-4)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        inputs, targets = batch['image'], batch['targets']
        total_loss, loss_array = self.detector([inputs, targets])
        
        self.log_dict({"total_loss" : total_loss, "box_loss" : loss_array[0], "cls_loss" : loss_array[1], "dfl_loss" : loss_array[2]}, on_epoch=True, batch_size=1, logger=True, reduce_fx='mean')
        self.log("debug_loss", total_loss, prog_bar=True, on_step=True)
        
        return total_loss
  
    def on_train_epoch_end(self) -> None:  
        return 
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch['image'], batch['targets']
        total_loss, loss_array = self.detector([inputs, targets])
        self.log('val_loss', total_loss, batch_size=1, on_epoch=True)
    
    def infer_batch(self, batch):
        
        return 

