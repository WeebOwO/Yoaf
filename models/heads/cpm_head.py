from ..block import *

class ClsRegHead(nn.Module):
    def __init__(self, in_channels, feature_size=96, conv_num=2,
                 norm_type='groupnorm', act_type='LeakyReLU'):
        super(ClsRegHead, self).__init__()
        
        conv_s = []
        for i in range(conv_num):
            if i == 0:
                conv_s.append(
                    ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
            else:
                conv_s.append(
                    ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
        self.conv_s = nn.Sequential(*conv_s)
        self.cls_output = nn.Conv3d(feature_size, 1, kernel_size=3, padding=1)

        conv_r = []
        for i in range(conv_num):
            if i == 0:
                conv_r.append(
                    ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
            else:
                conv_r.append(
                    ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
        self.conv_r = nn.Sequential(*conv_r)
        self.shape_output = nn.Conv3d(feature_size, 1, kernel_size=3, padding=1)
        
        conv_o = []
        for i in range(conv_num):
            if i == 0:
                conv_o.append(
                    ConvBlock(in_channels, feature_size, 3, norm_type=norm_type, act_type=act_type))
            else:
                conv_o.append(
                    ConvBlock(feature_size, feature_size, 3, norm_type=norm_type, act_type=act_type))
        self.conv_o = nn.Sequential(*conv_o)
        self.offset_output = nn.Conv3d(feature_size, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        shape = self.shape_output(self.conv_r(x))
        offset = self.offset_output(self.conv_o(x))
        cls = self.cls_output(self.conv_s(x))
    
        return cls, offset, shape
    
        
