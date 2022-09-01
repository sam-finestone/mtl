import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from resnet_encoder import Block
from models.attention.attention import LocalContextAttentionBlock
from sync_batchnorm.batchnorm import SynchronizedBatchNorm1d, \
    DataParallelWithCallback, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from torch.distributions.uniform import Uniform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

class SegDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, drop_out, BatchNorm):
        super(SegDecoder, self).__init__()
        self.input_height = input_dim
        self.conv1 = nn.Conv2d(input_dim, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
        )
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=(128, 256), mode="bilinear", align_corners=True)
        x = self.last_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DepthDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, drop_out, BatchNorm):
        super(DepthDecoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1),
        )
        # self.softplus = torch.nn.Softplus(beta=1, threshold=20)
        self.sigmoid = torch.nn.Sigmoid()
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = torch.cat((x, low_level_feat), dim=1)
        x = F.interpolate(x, size=(128, 256), mode="bilinear", align_corners=True)
        x = self.last_conv(x)
        # x = self.softplus(x)
        # x = self.sigmoid(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MultiDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, drop_out, BatchNorm):
        super(MultiDecoder, self).__init__()
        # Segmentation
        self.seg_decoder = SegDecoder(input_dim, num_classes[1], drop_out, BatchNorm)
        # Depth
        self.depth_decoder = DepthDecoder(input_dim, num_classes[0], drop_out, BatchNorm)
        self._init_weight()

    def forward(self, x):
        # compute the depth
        depth_pred = self.depth_decoder(x)
        # compute the segmentation
        seg_pred = self.seg_decoder(x)
        return depth_pred, seg_pred

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DecoderTemporal(nn.Module):
    def __init__(self, input_c_dim, num_classes, drop_out, BatchNorm):
        super(DecoderTemporal, self).__init__()
        self.input_c_dim = input_c_dim
        self.num_classes = num_classes
        mid_input_dim = 128
        self.conv1 = nn.Conv3d(input_c_dim, mid_input_dim,  kernel_size=(3, 1, 1), bias=False)
        self.bn1 = BatchNorm(mid_input_dim)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            # nn.Conv3d(mid_input_dim, mid_input_dim, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
            # BatchNorm(mid_input_dim),
            # nn.ReLU(),
            # nn.Dropout(drop_out),
            nn.Conv3d(mid_input_dim, mid_input_dim, kernel_size=(3, 3, 3), stride=1, padding=1, bias=False),
            BatchNorm(mid_input_dim),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Conv3d(mid_input_dim, self.num_classes, kernel_size=(1, 1, 1), stride=1),
        )
        self._init_weight()

    def forward(self, x):
        # permute the input tensor of dim [B, T, C, H, W] to [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        # print('input ,x, before first conv layer in Segdecotemp')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = torch.cat((x, low_level_feat), dim=1)
        x = F.interpolate(x, size=(1, 128, 256), mode="trilinear", align_corners=True)
        # x = F.interpolate(x, size=(128, 256), mode="bilinear", align_corners=True)
        x = self.last_conv(x)
        x = x.squeeze()  # use if using trilinear interpolation
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SE_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


######################### Semi-supervision decoders ##################################

class DropOutDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, drop_out, BatchNorm, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__()
        self.dropout = nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        self.seg_decoder = SegDecoder(input_dim, num_classes, drop_out, BatchNorm).to(device)

    def forward(self, x):
        x = self.dropout(x)
        x = self.seg_decoder(x)
        return x


class FeatureDropDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, drop_out, BatchNorm):
        super(FeatureDropDecoder, self).__init__()
        self.seg_decoder = SegDecoder(input_dim, num_classes, drop_out, BatchNorm).to(device)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x):
        x = self.feature_dropout(x)
        x = self.seg_decoder(x)
        return x


class FeatureNoiseDecoder(nn.Module):
    def __init__(self, input_dim, num_classes, drop_out, BatchNorm, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__()
        self.seg_decoder = SegDecoder(input_dim, num_classes, drop_out, BatchNorm).to(device)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        x = self.seg_decoder(x)
        return x


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3, 3), dilation=(1, 1, 1), bias=False):
        super().__init__()
        assert len(kernel_size) == 3, 'kernel_size must be a 3-tuple.'
        time_pad = (kernel_size[0] - 1) * dilation[0]
        height_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        width_pad = ((kernel_size[2] - 1) * dilation[2]) // 2

        # Pad temporally on the left
        self.pad = nn.ConstantPad3d(padding=(width_pad, width_pad, height_pad, height_pad, time_pad, 0), value=0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, dilation=dilation, stride=1, padding=0, bias=bias)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

        w = torch.ones_like(self.conv.weight)
        self.conv.weight = torch.nn.Parameter(w)

    def forward(self, *inputs):
        (x,) = inputs
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


