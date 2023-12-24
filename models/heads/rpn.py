from ..block import *

# simple rpn head without norm
class RpnHead(nn.Module):
    def __init__(self, input_chan=96) -> None:
        super(RpnHead, self).__init__()

        self.hm_head = nn.Sequential(
            nn.Conv3d(input_chan, input_chan, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_chan, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        )

        self.radius_head = nn.Sequential(
            nn.Conv3d(input_chan, input_chan, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_chan, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.reg_head = nn.Sequential(
            nn.Conv3d(input_chan, input_chan, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_chan, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1)) 
        ) 

    def forward(self, feature):
        return self.hm_head(feature), self.reg_head(feature), self.radius_head(feature)