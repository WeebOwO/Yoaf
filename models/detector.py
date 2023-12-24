import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L

from .heads import *
from .backbone import *

from config import net_config

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
        
    def decode(self, hm_feature, reg_feature, rad_feature, confidence, stride):
        hmax = F.max_pool3d(hm_feature, kernel_size=3, stride=1, padding=1) # perform max pooling to generate result
        keep = (hmax == hm_feature).float()
        
        hm_feature = keep * hm_feature
        pred = hm_feature.squeeze()
        reg = reg_feature.squeeze().permute(1, 2, 3, 0)

        rad = torch.exp(rad_feature.squeeze())
        D, H, W = pred.shape

        oz = torch.arange(0, D, 1).cuda()
        oh = torch.arange(0, H, 1).cuda()
        ow = torch.arange(0, W, 1).cuda()
        bbox = torch.ones((D, H, W, 5)).cuda()

        bbox[:, :, :, 0] = pred
        bbox[:, :, :, 1] = (reg[:, :, :, 0] + oz.reshape(-1, 1, 1)) * stride
        bbox[:, :, :, 2] = (reg[:, :, :, 1] + oh.reshape(1, -1, 1)) * stride
        bbox[:, :, :, 3] = (reg[:, :, :, 2] + ow.reshape(1, 1, -1)) * stride
        bbox[:, :, :, 4] = rad
        
        bbox = bbox.reshape(-1, 5)
        bbox = bbox[bbox[:, 0] > confidence]
        return bbox
    
    def inference(self, x):
        feature = self.backbone(x)
        stride = int(x.shape[-1] / feature.shape[-1])
        hm_feature, reg_feature, rad_feature = self.head(feature)
        pred = torch.sigmoid(hm_feature)
        
        bboxs = self.decode(pred, reg_feature, rad_feature, 0.3, stride)
        bboxs = bboxs[torch.argsort(-bboxs[:,0])][:net_config['infer_topk']]
        
        return bboxs.cpu().numpy()
        
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
        
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.detector.parameters(), self.lr or self.learning_rate, momentum=self.momentum, nesterov=True)
        lr_warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters=self.warm_up)
        cos_lr = CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=1e-6)
        return [optimizer], [lr_warmup, cos_lr]
            
    def training_step(self, batch, batch_idx):
        # bs = len(batch[0])
        optimizer = self.optimizers()
        inputs, targets = batch['image'], batch['targets']
        total_loss, loss_array = self.detector([inputs, targets])
        
        self.log_dict({"total_loss" : total_loss, "box_loss" : loss_array[0], "cls_loss" : loss_array[1], "dfl_loss" : loss_array[2]}, on_epoch=True, batch_size=1, logger=True, reduce_fx='mean')
        self.log("debug_loss", total_loss, prog_bar=True, on_step=True)
        
        optimizer.zero_grad()
        self.manual_backward(total_loss)
        optimizer.step()
  
    def on_train_epoch_end(self) -> None:  
        lr_warmup, lr_cos = self.lr_schedulers()
        if self.trainer.current_epoch < self.warm_up:
            lr_warmup.step()
        else:
            lr_cos.step()
        
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch['image'], batch['targets']
        _, loss_array = self.detector([inputs, targets])
        self.log('val_loss', loss_array[0], batch_size=1)
    
    def infer_batch(self, batch):
        
        return 

