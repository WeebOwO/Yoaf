import torch 
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from ..block import DFL

class TaskDecomposition(nn.Module):
    def __init__(self, feat_channels, stack_convs, la_down_rate=8):
        super(TaskDecomposition, self).__init__()
        self.feature_chan = feat_channels
        self.stack_convs = stack_convs
        self.in_channels = self.stack_convs * self.feature_chan
        
        self.la_conv1 = nn.Conv3d(self.in_channels, self.in_channels // la_down_rate, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.la_conv2 = nn.Conv3d(self.in_channels // la_down_rate, self.stack_convs, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        
        self.reduction_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv3d(self.in_channels, self.feature_chan, kernel_size=1, stride=1, padding=0)),
            ('norm', nn.GroupNorm(num_groups=self.feature_chan // 8, num_channels=self.feature_chan)),
            ('act',  nn.LeakyReLU())]
        ))
        
    def forward(self, feat, avg_feat=None):
        b, c, d, h, w = feat.shape
        if avg_feat == None:
            F.adaptive_avg_pool3d(feat, (1, 1, 1))
        
        weight = self.act(self.la_conv1(avg_feat))
        weight = self.sigmoid(self.la_conv2(weight))
        
        conv_weight = weight.reshape(b, 1, self.stack_convs, 1) * self.reduction_conv.conv.weight.reshape(1, self.feature_chan, self.stack_convs, self.feature_chan)
        conv_weight = conv_weight.reshape(b, self.feature_chan, self.in_channels)
        
        feat = feat.reshape(b, self.in_channels, d * h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feature_chan, d, h, w)
        
        return self.reduction_conv.act(self.reduction_conv.norm(feat))
    
class TOOD(nn.Module):
    def __init__(self, cls=1, in_channels=96, stack_convs=2, feature_chan=96, reg_max=8) -> None:
        super().__init__()
        self.cls = cls
        self.in_channels = in_channels
        self.stacked_convs = stack_convs
        self.feat_channels = feature_chan
        
        self.reg_max = reg_max
        self._init_layers()
        
    def _init_layers(self):
        self.inner_convs = nn.ModuleList()
        
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.inner_convs.append(
                nn.Sequential(
                    nn.Conv3d(in_channels=chn, out_channels=self.feat_channels, kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(num_groups=self.feat_channels // 8, num_channels=self.feat_channels),
                    nn.LeakyReLU()
                )
            )
        
        # task conv
        self.cls_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8)
        self.reg_decomp = TaskDecomposition(self.feat_channels, self.stacked_convs, self.stacked_convs * 8)
        
        self.tood_cls = nn.Conv3d(in_channels=self.feat_channels, out_channels=self.cls, kernel_size=1, stride=1) # record score
        self.tood_reg = nn.Conv3d(in_channels=self.feat_channels, out_channels=self.reg_max * 6, kernel_size=1, stride=1) # record reg max
    
        self.dfl = DFL(self.reg_max)
    
    def forward(self, x):
        bs = x.shape[0]
        inner_features = []
        
        for _, inner_conv in enumerate(self.inner_convs):
            x = inner_conv(x)
            inner_features.append(x)
        
        feat = torch.cat(inner_features, 1)
        
        avg_feat = F.adaptive_avg_pool3d(feat, (1, 1, 1))
        
        cls_feat = self.cls_decomp(feat, avg_feat)
        reg_feat = self.reg_decomp(feat, avg_feat)
        
        cls_feat = self.tood_cls(cls_feat)
        reg_feat = self.tood_reg(reg_feat)
        
        return [cls_feat, reg_feat] 
    
if __name__ == '__main__':
    tood = TOOD()
    feature = torch.rand(1, 96, 32, 32, 32)
    out = tood(feature)