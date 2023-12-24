from ..block import *

class Unet(nn.Module):
    def __init__(self, n_channels=1, n_blocks=[2, 3, 3, 3], n_filters=[64, 96, 128, 160], stem_filters=32,
                 norm_type='batchnorm', act_type='LeakyReLU', coord=True, first_stride=(2, 2, 2)):
        super(Unet, self).__init__()

        self.in_conv = ConvBlock(n_channels, stem_filters, stride=1, norm_type=norm_type, act_type=act_type)
        self.in_dw = ConvBlock(stem_filters, n_filters[0], stride=first_stride, norm_type=norm_type, act_type=act_type)

        self.block1 = LayerBasic(n_blocks[0], n_filters[0], n_filters[0], norm_type=norm_type, act_type=act_type, coord=coord)
        self.block1_dw = DownsamplingConvBlock(n_filters[0], n_filters[1], norm_type=norm_type, act_type=act_type)

        self.block2 = LayerBasic(n_blocks[1], n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type, coord=coord)
        self.block2_dw = DownsamplingConvBlock(n_filters[1], n_filters[2], norm_type=norm_type, act_type=act_type)

        self.block3 = LayerBasic(n_blocks[2], n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type, coord=coord)
        self.block3_dw = DownsamplingConvBlock(n_filters[2], n_filters[3], norm_type=norm_type, act_type=act_type)

        self.block4 = LayerBasic(n_blocks[3], n_filters[3], n_filters[3], norm_type=norm_type, act_type=act_type, coord=coord)

        self.block33_up = UpsamplingDeconvBlock(n_filters[3], n_filters[2], norm_type=norm_type, act_type=act_type)
        self.block33_res = LayerBasic(1, n_filters[2], n_filters[2], norm_type=norm_type, act_type=act_type, coord=coord)
        self.block33 = LayerBasic(2, n_filters[2] * 2, n_filters[2], norm_type=norm_type, act_type=act_type, coord=coord)

        self.block22_up = UpsamplingDeconvBlock(n_filters[2], n_filters[1], norm_type=norm_type, act_type=act_type)
        self.block22_res = LayerBasic(1, n_filters[1], n_filters[1], norm_type=norm_type, act_type=act_type, coord=coord)
        self.block22 = LayerBasic(2, n_filters[1] * 2, n_filters[1], norm_type=norm_type, act_type=act_type, coord=coord)
    
    def forward(self, x):
        # forward path in ResUnet
        x = self.in_conv(x)
        x = self.in_dw(x)

        x1 = self.block1(x)
        x = self.block1_dw(x1)

        x2 = self.block2(x)
        x = self.block2_dw(x2)

        x3 = self.block3(x)
        x = self.block3_dw(x3)

        x = self.block4(x)

        # back path in ResUnet
        x = self.block33_up(x)
        x3 = self.block33_res(x3)
        x = torch.cat([x, x3], dim=1)
        x = self.block33(x)

        x = self.block22_up(x)
        x2 = self.block22_res(x2)
        x = torch.cat([x, x2], dim=1)
        x = self.block22(x)

        return x
