import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_encoder import Block
from torch.distributions.categorical import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.transforms import CenterCrop

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class ILA_Module(nn.Module):
    '''
    This class contains the inter-frame local attention (ILA)
    which accounts for motion by finding local attention
    weights in inter-frames
    Agrs:
        L: the window size
    '''
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.conv1 = nn.Conv2d(3, 3, 3)
        # self.shared_weight = nn.Parameter(self.conv1.weight)



    def forward(self, ft, fk):
        start = -self.L/2
        end = self.L/2
        # For each pixel on the feature map fk, we propagate the features
        # from ft based on a weighted combination of pixels in a local neighborhood.
        # for x in range(start, end):
        #     for y in range(start, end):
        #         W_ij =
        pass


class Decoder(nn.Module):
    ''' This class contains the implementation of the inter-frame attention Module.
        Args:
            output_dim: A integer indicating the size of output dimension.
            hidden_dim: A integer indicating the hidden size of rnn.
            n_layers: A integer indicating the number of layers in rnn.
            dropout: A float indicating the dropout.
        '''

    def __init__(self, input_dim, chs=(2048, 512, 256)):
        super().__init__()
        # SE block - input encoder dimension and reduction ratio
        self.se_block = SE_Block(3, 16)
        # self.output_dim = output_dim
        # self.conv1 = nn.Conv3d(input_dim, 256, kernel_size=(5, 1, 1), stride=(1, 2, 2), bias=False)
        # self.conv2 = nn.Conv3d(256, 512, kernel_size=(5, 1, 1), stride=(1, 2, 2), bias=False)
        # self.conv3 = nn.Conv3d(128, 64, kernel_size=(5, 1, 1), stride=(1, 2, 2), bias=False)
        # self.chs = chs
        # self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        # self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.conv1 = nn.Conv2d(1280, 512, kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=(1, 1), stride=2)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_trans = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=2, stride=2)

    def forward(self, x):
        # for i in range(len(self.chs) - 1):
        #     x = self.upconvs[i](x)
        #     enc_ftrs = self.crop(encoder_features[i], x)
        #     x = torch.cat([x, enc_ftrs], dim=1)
        #     x = self.dec_blocks[i](x)
        # return x
        # x = self.se_block(x)
        # print(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.upsample(x)
        # x = self.up_trans(x)
        print(x.size())

    # def crop(self, enc_ftrs, x):
    #     _, _, H, W = x.shape
    #     enc_ftrs = CenterCrop([H, W])(enc_ftrs)
    #     return enc_ftrs
     # output based on task


    # def forward(self, enocoder_output, prev_fast_sblock, keyframe):
    #     # create an SE Block for each task - segmentation
    #     print(enocoder_output.shape)
    #     # fk = self.se_block(enocoder_output)
    #
    #
    #     return 0

