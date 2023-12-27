from ..block import *

class ClsRegHead(nn.Module):
    def __init__(self, in_channels, feature_size=96, conv_num=2, cls=1, reg_max=8,
                 norm_type='GroupNorm', act_type='LeakyReLU'):
        super(ClsRegHead, self).__init__()
        
        self.cls = cls
        self.reg_max = reg_max
        self.dfl = DFL(self.reg_max)
        
        conv_s = []
        for i in range(conv_num):
            if i == 0:
                conv_s.append(
                    ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
            else:
                conv_s.append(
                    ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
        self.conv_s = nn.Sequential(*conv_s)
        self.cls_head = nn.Conv3d(feature_size, cls, kernel_size=3, padding=1)
        
        conv_o = []
        for i in range(conv_num):
            if i == 0:
                conv_o.append(
                    ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
            else:
                conv_o.append(
                    ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
                
        self.conv_o = nn.Sequential(*conv_o)
        self.reg_head = nn.Conv3d(feature_size, self.reg_max * 6, kernel_size=3, padding=1)
        
    def forward(self, x):
        return [self.cls_head(x), self.reg_head(x)]
    
        
