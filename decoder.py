import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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


class Decoder(nn.Module):
    ''' This class contains the implementation of the inter-frame attention Module.
        Args:
            output_dim: A integer indicating the size of output dimension.
            hidden_dim: A integer indicating the hidden size of rnn.
            n_layers: A integer indicating the number of layers in rnn.
            dropout: A float indicating the dropout.
        '''

    def __init__(self, output_dim, input_dim, L):
        super().__init__()
        # SE block - input encoder dimension and reduction ratio
        self.se_block = SE_Block(3, 16)
        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(input_dim, 256, kernel_size=(5, 1, 1), stride=(1, 2, 2), bias=False)
        self.conv2 = nn.Conv3d(256, 512, kernel_size=(5, 1, 1), stride=(1, 2, 2), bias=False)
        self.conv3 = nn.Conv3d(128, 64, kernel_size=(5, 1, 1), stride=(1, 2, 2), bias=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
         # output based on task


    def forward(self, prev_sblock_kf, current_se_block_outputs, task=0):
        x = torch.cat((prev_sblock_kf, current_se_block_outputs), 1)
        # apply conv1
        x1 = F.relu(self.conv1(x))
        # apply conv2
        x2 = F.relu(self.conv2(x1))
        # apply conv3
        x3 = F.relu(self.conv2(x2))
        # apply the Uupsampling
        x4 = self.upsample(x3)
        # x = F.max_pool2d(x, 2)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # output = F.log_softmax(x, dim=1)
        return x4

